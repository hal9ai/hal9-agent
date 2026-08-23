"""
Clone a GitHub repository, apply changes with Claude Code, and open a PR.

Uses GITHUB_PAT for git/GitHub API auth and ANTHROPIC_API_KEY for Claude.
The repo is cloned into a system temp directory (not ./.storage) so it is
never uploaded to Hal9 via h9.save(). Changes are pushed on a feature
branch and opened as a pull request targeting main.
"""

import asyncio
import base64
import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import zipfile
from io import BytesIO
from typing import Optional
from urllib.parse import urlparse

import hal9 as h9
import requests


BASE_BRANCH = "main"
DEFAULT_MODEL = "sonnet"
DEFAULT_COMMIT_MESSAGE = "Apply changes via Hal9 Claude Code"


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(
            f"Missing required environment variable '{name}'. "
            f"Set {name} before using this tool."
        )
    return value


def _parse_github_repo(repo: str) -> tuple[str, str]:
    """
    Parse owner/repo from common GitHub URL or shorthand forms.

    Accepts:
      - owner/repo
      - https://github.com/owner/repo
      - https://github.com/owner/repo.git
      - git@github.com:owner/repo.git
    """
    repo = repo.strip()

    # owner/repo shorthand
    if re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        owner, name = repo.split("/", 1)
        return owner, name.removesuffix(".git")

    # git@github.com:owner/repo.git
    ssh_match = re.match(r"git@github\.com:([\w.-]+)/([\w.-]+?)(?:\.git)?$", repo)
    if ssh_match:
        return ssh_match.group(1), ssh_match.group(2)

    # https://github.com/owner/repo(.git)
    parsed = urlparse(repo)
    if parsed.netloc in ("github.com", "www.github.com") and parsed.path:
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(parts) >= 2:
            return parts[0], parts[1].removesuffix(".git")

    raise ValueError(
        f"Could not parse GitHub repository from '{repo}'. "
        "Use 'owner/repo' or a full github.com URL."
    )


def _authenticated_clone_url(owner: str, name: str, token: str) -> str:
    return f"https://x-access-token:{token}@github.com/{owner}/{name}.git"


def _redact_token(text: str) -> str:
    return re.sub(r"x-access-token:[^@\s]+@", "x-access-token:***@", text)


def _log_git(message: str) -> None:
    print(f"[git] {message}", flush=True)
    try:
        h9.event("Claude Code git", message[:500])
    except Exception:
        pass


def _clip(text: str, limit: int = 1200) -> str:
    text = (text or "").strip()
    if len(text) > limit:
        return text[:limit] + f"... ({len(text)} chars)"
    return text


def _is_root() -> bool:
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:
        return False
    return geteuid() == 0


def _privileged_variants(cmd: list[str], allow_sudo: bool = True) -> list[list[str]]:
    variants = [cmd]
    if allow_sudo and not _is_root() and shutil.which("sudo"):
        variants.append(["sudo", "-n", *cmd])
    return variants


def _run_install_command(cmd: list[str], env: Optional[dict] = None) -> subprocess.CompletedProcess:
    _log_git(f"running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=5 * 60,
    )
    _log_git(f"exit {result.returncode}")
    if result.stdout:
        _log_git(f"stdout: {_clip(result.stdout)}")
    if result.stderr:
        _log_git(f"stderr: {_clip(result.stderr)}")
    return result


def _known_git_paths() -> list[str]:
    return [
        shutil.which("git") or "",
        "/usr/bin/git",
        "/usr/local/bin/git",
        "/bin/git",
        "/opt/homebrew/bin/git",
    ]


def _find_git() -> Optional[str]:
    for path in _known_git_paths():
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _log_git_diagnostics() -> None:
    uid = getattr(os, "getuid", lambda: "n/a")()
    euid = getattr(os, "geteuid", lambda: "n/a")()
    _log_git(f"uid={uid} euid={euid} root={_is_root()} cwd={os.getcwd()}")
    _log_git(f"PATH={os.environ.get('PATH', '')}")
    for tool in (
        "git", "sudo", "apt-get", "apt", "apk", "dnf", "yum",
        "microdnf", "pacman", "zypper", "brew",
    ):
        _log_git(f"which {tool} = {shutil.which(tool) or '(not found)'}")
    for path in ("/usr/bin/git", "/usr/local/bin/git", "/bin/git"):
        exists = os.path.exists(path)
        executable = os.access(path, os.X_OK) if exists else False
        _log_git(f"{path} exists={exists} executable={executable}")


def _ensure_git() -> Optional[str]:
    """
    Return the git executable path if available, installing it when possible.
    Returns None when git cannot be installed (e.g. no root); callers should
    fall back to the GitHub API.
    """
    existing = _find_git()
    if existing:
        _log_git(f"found git at {existing}")
        return existing

    _log_git("git not found; attempting system install")
    _log_git_diagnostics()

    strategies: list[tuple[str, list[list[str]], Optional[dict], bool]] = []
    apt_env = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
    if shutil.which("apt-get"):
        strategies.append((
            "apt-get",
            [
                ["apt-get", "update", "-qq"],
                ["apt-get", "install", "-y", "--no-install-recommends", "git"],
            ],
            apt_env,
            True,
        ))
    if shutil.which("apk"):
        strategies.append(("apk", [["apk", "add", "--no-cache", "git"]], None, True))
    if shutil.which("dnf"):
        strategies.append(("dnf", [["dnf", "install", "-y", "git"]], None, True))
    if shutil.which("yum"):
        strategies.append(("yum", [["yum", "install", "-y", "git"]], None, True))
    if shutil.which("microdnf"):
        strategies.append(("microdnf", [["microdnf", "install", "-y", "git"]], None, True))
    if shutil.which("pacman"):
        strategies.append(("pacman", [["pacman", "-Sy", "--noconfirm", "git"]], None, True))
    if shutil.which("zypper"):
        strategies.append(("zypper", [["zypper", "--non-interactive", "install", "git"]], None, True))
    if shutil.which("brew"):
        strategies.append(("brew", [["brew", "install", "git"]], None, False))

    if not strategies:
        _log_git("no supported package manager found on PATH")

    for name, commands, env, allow_sudo in strategies:
        _log_git(f"trying installer: {name}")
        last_error = ""
        installed = True
        for cmd in commands:
            step_ok = False
            for variant in _privileged_variants(cmd, allow_sudo=allow_sudo):
                try:
                    result = _run_install_command(variant, env=env)
                except Exception as e:
                    last_error = str(e)
                    _log_git(f"command failed to start: {last_error}")
                    continue
                if result.returncode == 0:
                    step_ok = True
                    break
                last_error = (result.stderr or result.stdout or "").strip()
            if not step_ok:
                if cmd and cmd[0] == "apt-get" and "update" in cmd:
                    _log_git("apt-get update failed; still trying apt-get install")
                    continue
                installed = False
                break
        if not installed:
            _log_git(f"{name} failed: {_clip(last_error)}")
            continue
        git_path = _find_git()
        if git_path:
            _log_git(f"installed git via {name}: {git_path}")
            return git_path
        _log_git(f"{name} reported success but git is still not on PATH")

    _log_git("could not install git; will use GitHub HTTP API instead")
    return None


def _git_bin() -> str:
    path = _ensure_git()
    if not path:
        raise RuntimeError("git is not installed")
    return path


def _run_git(args: list[str], cwd: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [_git_bin(), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        stderr = _redact_token((result.stderr or result.stdout or "").strip())
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr}")
    return result


def _clone_base_and_checkout_branch(clone_url: str, dest: str, base_branch: str, work_branch: str) -> None:
    """Clone `base_branch` (usually main), then create the feature branch from it."""
    git = _git_bin()
    result = subprocess.run(
        [git, "clone", "--depth", "1", "--branch", base_branch, clone_url, dest],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        result = subprocess.run(
            [git, "clone", "--depth", "1", clone_url, dest],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = _redact_token((result.stderr or result.stdout or "").strip())
            raise RuntimeError(f"Failed to clone repository: {stderr}")

    _run_git(["checkout", "-B", work_branch], cwd=dest)


def _slugify_branch(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
    slug = slug[:40].strip("-") or "changes"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"hal9/{slug}-{stamp}"


def _normalize_work_branch(branch: str, prompt: str) -> str:
    branch = (branch or "").strip()
    if not branch or branch in ("main", "master", BASE_BRANCH):
        return _slugify_branch(prompt)
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", branch):
        return _slugify_branch(prompt)
    return branch.lstrip("/")


def _collect_text_from_sdk_message(message) -> list[str]:
    texts: list[str] = []
    try:
        from claude_agent_sdk import AssistantMessage, TextBlock, ResultMessage
    except ImportError:
        return texts

    if isinstance(message, AssistantMessage):
        for block in getattr(message, "content", []) or []:
            if isinstance(block, TextBlock) and getattr(block, "text", None):
                texts.append(block.text)
    elif isinstance(message, ResultMessage):
        result = getattr(message, "result", None)
        if result:
            texts.append(str(result))
    return texts


async def _run_claude_agent_sdk(prompt: str, cwd: str, model: str, api_key: str) -> str:
    from claude_agent_sdk import query, ClaudeAgentOptions

    options_kwargs = {
        "cwd": cwd,
        "model": model,
        "permission_mode": "bypassPermissions",
        "allowed_tools": ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "NotebookEdit"],
        "max_turns": 50,
        "env": {"ANTHROPIC_API_KEY": api_key},
        "system_prompt": (
            "You are editing a cloned git repository to fulfill the user's request. "
            "Make the necessary code and file changes. Do not commit, push, or modify "
            "git remotes — git operations after your edits are handled externally."
        ),
    }

    # Some SDK versions require an explicit safety flag for bypassPermissions.
    try:
        options = ClaudeAgentOptions(
            **options_kwargs,
            allow_dangerously_skip_permissions=True,
        )
    except TypeError:
        options = ClaudeAgentOptions(**options_kwargs)

    collected: list[str] = []
    async for message in query(prompt=prompt, options=options):
        collected.extend(_collect_text_from_sdk_message(message))

    return "\n".join(collected).strip() or "Claude Code finished without a text summary."


def _run_claude_cli(prompt: str, cwd: str, model: str, api_key: str) -> str:
    """Fallback: invoke the Claude Code CLI non-interactively."""
    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = api_key

    cmd = [
        "claude",
        "--bare",
        "-p",
        prompt,
        "--model",
        model,
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        "Read,Write,Edit,Bash,Glob,Grep,NotebookEdit",
        "--append-system-prompt",
        (
            "You are editing a cloned git repository to fulfill the user's request. "
            "Make the necessary code and file changes. Do not commit, push, or modify "
            "git remotes — git operations after your edits are handled externally."
        ),
    ]

    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
        timeout=60 * 30,  # 30 minutes
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "unknown CLI error").strip()
        raise RuntimeError(f"Claude Code CLI failed (exit {result.returncode}): {err}")

    return (result.stdout or "").strip() or "Claude Code CLI finished without output."


def _run_claude_code(prompt: str, cwd: str, model: str, api_key: str) -> str:
    try:
        import claude_agent_sdk  # noqa: F401
        h9.event("Claude Code", "Using claude-agent-sdk Python package")
        return asyncio.run(_run_claude_agent_sdk(prompt, cwd, model, api_key))
    except ImportError:
        h9.event("Claude Code", "claude-agent-sdk not installed; falling back to CLI")
    except Exception as e:
        h9.event("Claude Code SDK error", str(e))
        # Fall through to CLI if SDK is present but failed in a recoverable way
        if shutil.which("claude") is None:
            raise

    if shutil.which("claude") is None:
        raise RuntimeError(
            "Neither the 'claude-agent-sdk' Python package nor the 'claude' CLI is available. "
            "Install with: pip install claude-agent-sdk"
        )
    return _run_claude_cli(prompt, cwd, model, api_key)


def _has_changes(cwd: str) -> bool:
    result = _run_git(["status", "--porcelain"], cwd=cwd)
    return bool(result.stdout.strip())


def _commits_ahead_of_base(cwd: str, base_branch: str) -> int:
    result = _run_git(
        ["rev-list", "--count", f"origin/{base_branch}..HEAD"],
        cwd=cwd,
        check=False,
    )
    if result.returncode != 0:
        result = _run_git(
            ["rev-list", "--count", f"{base_branch}..HEAD"],
            cwd=cwd,
            check=False,
        )
    try:
        return int((result.stdout or "0").strip() or "0")
    except ValueError:
        return 0


def _commit_changes(cwd: str, commit_message: str) -> str:
    _run_git(["config", "user.email", "hal9@hal9.com"], cwd=cwd)
    _run_git(["config", "user.name", "Hal9 Claude Code"], cwd=cwd)
    _run_git(["add", "-A"], cwd=cwd)
    if not _has_changes(cwd):
        return ""
    commit = _run_git(["commit", "-m", commit_message], cwd=cwd)
    return (commit.stdout or "").strip()


def _push_branch(cwd: str, branch: str) -> str:
    push = _run_git(["push", "-u", "origin", f"HEAD:{branch}"], cwd=cwd)
    return (push.stdout or push.stderr or "").strip()


def _github_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


SKIP_DIR_NAMES = {".git", "__pycache__", ".venv", "node_modules", ".storage"}


def _github_json(method: str, url: str, token: str, **kwargs) -> requests.Response:
    _log_git(f"GitHub API {method} {url}")
    response = requests.request(
        method,
        url,
        headers=_github_headers(token),
        timeout=kwargs.pop("timeout", 60),
        **kwargs,
    )
    _log_git(f"GitHub API status {response.status_code}")
    return response


def _resolve_base_branch(owner: str, name: str, token: str) -> str:
    response = _github_json("GET", f"https://api.github.com/repos/{owner}/{name}", token, timeout=30)
    if response.status_code != 200:
        _log_git(f"repo lookup failed ({response.status_code}): {_clip(response.text)}")
        return BASE_BRANCH
    default_branch = (response.json() or {}).get("default_branch") or BASE_BRANCH
    main_ref = _github_json(
        "GET",
        f"https://api.github.com/repos/{owner}/{name}/git/ref/heads/{BASE_BRANCH}",
        token,
        timeout=30,
    )
    if main_ref.status_code == 200:
        return BASE_BRANCH
    _log_git(f"{BASE_BRANCH} ref missing; using default branch {default_branch}")
    return default_branch


def _clone_via_github_zip(owner: str, name: str, token: str, dest: str, base_branch: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{name}/zipball/{base_branch}"
    _log_git(f"downloading zipball {owner}/{name}@{base_branch}")
    response = requests.get(url, headers=_github_headers(token), timeout=120, allow_redirects=True)
    _log_git(f"zipball status {response.status_code} bytes={len(response.content)}")
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download {owner}/{name}@{base_branch} as zip "
            f"({response.status_code}): {_clip(response.text)}"
        )
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        archive.extractall(dest)
    entries = [os.path.join(dest, entry) for entry in os.listdir(dest)]
    dirs = [entry for entry in entries if os.path.isdir(entry)]
    repo_dir = dirs[0] if len(dirs) == 1 else dest
    _log_git(f"extracted repository to {repo_dir}")
    return repo_dir


def _snapshot_files(root: str) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            try:
                with open(path, "rb") as handle:
                    snapshot[rel] = hashlib.sha256(handle.read()).hexdigest()
            except OSError:
                continue
    return snapshot


def _diff_snapshots(before: dict[str, str], after: dict[str, str]) -> tuple[list[str], list[str]]:
    changed = sorted(
        path for path, digest in after.items()
        if before.get(path) != digest
    )
    deleted = sorted(path for path in before if path not in after)
    return changed, deleted


def _file_mode(path: str) -> str:
    mode = os.stat(path).st_mode
    if mode & stat.S_IXUSR:
        return "100755"
    return "100644"


def _create_blob(owner: str, name: str, token: str, content: bytes) -> str:
    response = _github_json(
        "POST",
        f"https://api.github.com/repos/{owner}/{name}/git/blobs",
        token,
        json={"content": base64.b64encode(content).decode("ascii"), "encoding": "base64"},
        timeout=60,
    )
    if response.status_code not in (200, 201):
        raise RuntimeError(f"Failed to create blob ({response.status_code}): {_clip(response.text)}")
    sha = (response.json() or {}).get("sha")
    if not sha:
        raise RuntimeError("GitHub blob response did not include a sha")
    return sha


def _publish_changes_via_github_api(
    owner: str,
    name: str,
    token: str,
    repo_dir: str,
    work_branch: str,
    base_branch: str,
    commit_message: str,
    before: dict[str, str],
) -> str:
    changed, deleted = _diff_snapshots(before, _snapshot_files(repo_dir))
    _log_git(f"file diff: {len(changed)} changed, {len(deleted)} deleted")
    if not changed and not deleted:
        return ""

    ref = _github_json(
        "GET",
        f"https://api.github.com/repos/{owner}/{name}/git/ref/heads/{base_branch}",
        token,
        timeout=30,
    )
    if ref.status_code != 200:
        raise RuntimeError(f"Failed to read {base_branch} ref ({ref.status_code}): {_clip(ref.text)}")
    base_commit_sha = ref.json()["object"]["sha"]

    commit = _github_json(
        "GET",
        f"https://api.github.com/repos/{owner}/{name}/git/commits/{base_commit_sha}",
        token,
        timeout=30,
    )
    if commit.status_code != 200:
        raise RuntimeError(f"Failed to read commit {base_commit_sha}: {_clip(commit.text)}")
    base_tree_sha = commit.json()["tree"]["sha"]

    tree_items = []
    for rel in changed:
        path = os.path.join(repo_dir, rel)
        with open(path, "rb") as handle:
            content = handle.read()
        _log_git(f"uploading blob for {rel} ({len(content)} bytes)")
        tree_items.append({
            "path": rel,
            "mode": _file_mode(path),
            "type": "blob",
            "sha": _create_blob(owner, name, token, content),
        })
    for rel in deleted:
        tree_items.append({
            "path": rel,
            "mode": "100644",
            "type": "blob",
            "sha": None,
        })

    tree = _github_json(
        "POST",
        f"https://api.github.com/repos/{owner}/{name}/git/trees",
        token,
        json={"base_tree": base_tree_sha, "tree": tree_items},
        timeout=60,
    )
    if tree.status_code not in (200, 201):
        raise RuntimeError(f"Failed to create tree ({tree.status_code}): {_clip(tree.text)}")

    new_commit = _github_json(
        "POST",
        f"https://api.github.com/repos/{owner}/{name}/git/commits",
        token,
        json={
            "message": commit_message,
            "tree": tree.json()["sha"],
            "parents": [base_commit_sha],
        },
        timeout=30,
    )
    if new_commit.status_code not in (200, 201):
        raise RuntimeError(
            f"Failed to create commit ({new_commit.status_code}): {_clip(new_commit.text)}"
        )
    new_sha = new_commit.json()["sha"]
    _log_git(f"created commit {new_sha} on {work_branch}")

    create_ref = _github_json(
        "POST",
        f"https://api.github.com/repos/{owner}/{name}/git/refs",
        token,
        json={"ref": f"refs/heads/{work_branch}", "sha": new_sha},
        timeout=30,
    )
    if create_ref.status_code in (200, 201):
        return new_sha
    if create_ref.status_code == 422:
        update_ref = _github_json(
            "PATCH",
            f"https://api.github.com/repos/{owner}/{name}/git/refs/heads/{work_branch}",
            token,
            json={"sha": new_sha, "force": True},
            timeout=30,
        )
        if update_ref.status_code in (200, 201):
            return new_sha
        raise RuntimeError(
            f"Failed to update branch {work_branch} ({update_ref.status_code}): {_clip(update_ref.text)}"
        )
    raise RuntimeError(
        f"Failed to create branch {work_branch} ({create_ref.status_code}): {_clip(create_ref.text)}"
    )


def _create_pull_request(
    owner: str,
    name: str,
    token: str,
    title: str,
    head: str,
    base: str,
    body: str,
) -> tuple[str, int]:
    url = f"https://api.github.com/repos/{owner}/{name}/pulls"
    response = requests.post(
        url,
        headers=_github_headers(token),
        json={"title": title, "head": head, "base": base, "body": body},
        timeout=30,
    )
    if response.status_code in (200, 201):
        data = response.json()
        return data.get("html_url") or "", int(data.get("number") or 0)

    # Reuse an existing open PR for the same head/base if one already exists.
    if response.status_code == 422:
        existing = requests.get(
            url,
            headers=_github_headers(token),
            params={"head": f"{owner}:{head}", "base": base, "state": "open"},
            timeout=30,
        )
        if existing.status_code == 200 and existing.json():
            data = existing.json()[0]
            return data.get("html_url") or "", int(data.get("number") or 0)

    detail = (response.text or "").strip()
    raise RuntimeError(f"Failed to create pull request ({response.status_code}): {detail}")


def claude_code_github(
    repo: str,
    prompt: str,
    branch: str = "",
    model: str = DEFAULT_MODEL,
    commit_message: str = DEFAULT_COMMIT_MESSAGE,
    pr_title: str = "",
):
    """
    Clone a GitHub repo, apply changes with Claude Code, push a feature branch,
    and open a pull request into main.
    """
    model = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    commit_message = (commit_message or DEFAULT_COMMIT_MESSAGE).strip() or DEFAULT_COMMIT_MESSAGE
    work_branch = _normalize_work_branch(branch, prompt)
    title = (pr_title or "").strip() or commit_message

    try:
        github_pat = _require_env("GITHUB_PAT")
        anthropic_key = _require_env("ANTHROPIC_API_KEY")
        owner, name = _parse_github_repo(repo)
    except ValueError as e:
        return f"Error: {e}"

    git_path = _ensure_git()
    clone_url = _authenticated_clone_url(owner, name, github_pat)
    public_url = f"https://github.com/{owner}/{name}"
    base_branch = BASE_BRANCH

    work_root: Optional[str] = None
    try:
        work_root = tempfile.mkdtemp(prefix="hal9-claude-code-")
        repo_dir = os.path.join(work_root, name)
        before_snapshot: Optional[dict[str, str]] = None

        if git_path:
            _log_git(f"cloning with git ({git_path})")
            h9.event("Claude Code", f"Cloning {public_url} ({base_branch}) into temp dir")
            _clone_base_and_checkout_branch(clone_url, repo_dir, base_branch, work_branch)
            _run_git(["remote", "set-url", "origin", clone_url], cwd=repo_dir)
        else:
            base_branch = _resolve_base_branch(owner, name, github_pat)
            _log_git(f"cloning via GitHub zip API (base={base_branch})")
            repo_dir = _clone_via_github_zip(owner, name, github_pat, repo_dir, base_branch)
            before_snapshot = _snapshot_files(repo_dir)

        h9.event("Claude Code", f"Running Claude Code (model={model}) on branch {work_branch}")
        claude_summary = _run_claude_code(prompt, repo_dir, model, anthropic_key)

        if git_path:
            commit_out = _commit_changes(repo_dir, commit_message)
            if _commits_ahead_of_base(repo_dir, base_branch) == 0:
                return (
                    f"Claude Code ran on {public_url} but made no file changes, so no pull request was opened.\n\n"
                    f"Claude summary:\n{claude_summary}"
                )
            h9.event("Claude Code", f"Pushing {work_branch} and opening PR into {base_branch}")
            push_out = _push_branch(repo_dir, work_branch)
        else:
            commit_out = _publish_changes_via_github_api(
                owner=owner,
                name=name,
                token=github_pat,
                repo_dir=repo_dir,
                work_branch=work_branch,
                base_branch=base_branch,
                commit_message=commit_message,
                before=before_snapshot or {},
            )
            if not commit_out:
                return (
                    f"Claude Code ran on {public_url} but made no file changes, so no pull request was opened.\n\n"
                    f"Claude summary:\n{claude_summary}"
                )
            push_out = f"Published commit {commit_out} via GitHub API"

        pr_body = (
            f"{prompt.strip()}\n\n"
            f"---\n"
            f"Opened automatically by Hal9 Claude Code.\n\n"
            f"Claude summary:\n{claude_summary}"
        )
        pr_url, pr_number = _create_pull_request(
            owner=owner,
            name=name,
            token=github_pat,
            title=title,
            head=work_branch,
            base=base_branch,
            body=pr_body,
        )

        return (
            f"Opened pull request #{pr_number} into {base_branch} for {public_url}.\n"
            f"PR: {pr_url}\n"
            f"Branch: {work_branch}\n\n"
            f"Commit: {commit_out}\n"
            f"Push: {push_out}\n\n"
            f"Claude summary:\n{claude_summary}"
        )
    except Exception as e:
        msg = _redact_token(str(e))
        h9.event("Claude Code error", msg)
        return f"Error while running Claude Code on GitHub repo: {msg}"
    finally:
        if work_root and os.path.isdir(work_root):
            shutil.rmtree(work_root, ignore_errors=True)


claude_code_github_description = {
    "type": "function",
    "function": {
        "name": "claude_code_github",
        "description": (
            "Clones a GitHub repository, uses Claude Code (ANTHROPIC_API_KEY) to make the "
            "requested code changes, pushes a feature branch, and opens a pull request into "
            "main using GITHUB_PAT. Use this when the user wants to change code in a GitHub "
            "repo and get a PR back. Never pushes directly to main."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": (
                        "GitHub repository as 'owner/repo' or a full github.com URL "
                        "(e.g. 'acme/api' or 'https://github.com/acme/api')."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "Detailed natural-language instructions for the changes Claude Code "
                        "should make in the repository."
                    ),
                },
                "branch": {
                    "type": "string",
                    "description": (
                        "Feature branch to push and open the PR from. Do not use 'main'. "
                        "If unsure, use a short name like 'hal9/update-readme'."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": (
                        "Claude model to use (e.g. 'sonnet', 'opus', 'haiku', or a full model id). "
                        "Use 'sonnet' unless the user requests a different model."
                    ),
                },
                "commit_message": {
                    "type": "string",
                    "description": (
                        "Git commit message for the changes. Provide a short, descriptive message; "
                        "if unsure use 'Apply changes via Hal9 Claude Code'."
                    ),
                },
                "pr_title": {
                    "type": "string",
                    "description": (
                        "Title for the GitHub pull request. Use a concise summary of the change."
                    ),
                },
            },
            "required": ["repo", "prompt", "branch", "model", "commit_message", "pr_title"],
            "additionalProperties": False,
        },
    },
}
