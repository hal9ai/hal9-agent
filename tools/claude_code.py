"""
Clone a GitHub repository, apply changes with Claude Code, and open a PR.

Uses GITHUB_PAT for git/GitHub API auth and ANTHROPIC_API_KEY for Claude.
The repo is cloned into a system temp directory (not ./.storage) so it is
never uploaded to Hal9 via h9.save(). Changes are pushed on a feature
branch and opened as a pull request targeting main.
"""

import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import time
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
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=5 * 60,
    )


def _ensure_git() -> str:
    """
    Return the git executable path, installing it via the system package
    manager when it is missing. pip cannot install the git CLI.
    """
    existing = shutil.which("git")
    if existing:
        return existing

    h9.event("Claude Code", "git not found; attempting to install it")

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

    errors: list[str] = []
    for name, commands, env, allow_sudo in strategies:
        last_error = ""
        installed = True
        for cmd in commands:
            step_ok = False
            for variant in _privileged_variants(cmd, allow_sudo=allow_sudo):
                result = _run_install_command(variant, env=env)
                if result.returncode == 0:
                    step_ok = True
                    break
                last_error = (result.stderr or result.stdout or "").strip()
            if not step_ok:
                # apt-get update can fail in locked environments; still try install.
                if cmd and cmd[0] == "apt-get" and "update" in cmd:
                    continue
                installed = False
                break
        if not installed:
            errors.append(f"{name}: {last_error}")
            continue
        git_path = shutil.which("git")
        if git_path:
            h9.event("Claude Code", f"Installed git via {name}: {git_path}")
            return git_path
        errors.append(f"{name}: command succeeded but git is still not on PATH")

    details = "; ".join(errors) if errors else "no supported package manager found"
    raise RuntimeError(
        "git is not installed and could not be installed automatically "
        f"({details}). Install git in the runtime image, or run this tool as root."
    )


def _git_bin() -> str:
    return _ensure_git()


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
        _ensure_git()
    except (ValueError, RuntimeError) as e:
        return f"Error: {e}"

    clone_url = _authenticated_clone_url(owner, name, github_pat)
    public_url = f"https://github.com/{owner}/{name}"

    work_root: Optional[str] = None
    try:
        work_root = tempfile.mkdtemp(prefix="hal9-claude-code-")
        repo_dir = os.path.join(work_root, name)

        h9.event("Claude Code", f"Cloning {public_url} ({BASE_BRANCH}) into temp dir")
        _clone_base_and_checkout_branch(clone_url, repo_dir, BASE_BRANCH, work_branch)
        _run_git(["remote", "set-url", "origin", clone_url], cwd=repo_dir)

        h9.event("Claude Code", f"Running Claude Code (model={model}) on branch {work_branch}")
        claude_summary = _run_claude_code(prompt, repo_dir, model, anthropic_key)

        commit_out = _commit_changes(repo_dir, commit_message)
        if _commits_ahead_of_base(repo_dir, BASE_BRANCH) == 0:
            return (
                f"Claude Code ran on {public_url} but made no file changes, so no pull request was opened.\n\n"
                f"Claude summary:\n{claude_summary}"
            )

        h9.event("Claude Code", f"Pushing {work_branch} and opening PR into {BASE_BRANCH}")
        push_out = _push_branch(repo_dir, work_branch)
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
            base=BASE_BRANCH,
            body=pr_body,
        )

        return (
            f"Opened pull request #{pr_number} into {BASE_BRANCH} for {public_url}.\n"
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
