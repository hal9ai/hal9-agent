"""
Change a GitHub repository with Claude Code and open a PR into main.

The Hal9 runtime has no git and cannot apt-install it (unprivileged).
This tool downloads the repo as a zip, lets Claude Code edit files locally,
then creates the branch, commit, and pull request through the GitHub HTTP API.
Uses GITHUB_PAT and ANTHROPIC_API_KEY. Work happens in a temp directory,
not ./.storage, so it is never uploaded via h9.save().
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


def _redact_token(text: str) -> str:
    return re.sub(r"x-access-token:[^@\s]+@", "x-access-token:***@", text)


def _clip(text: str, limit: int = 1200) -> str:
    text = (text or "").strip()
    if len(text) > limit:
        return text[:limit] + f"... ({len(text)} chars)"
    return text


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


def _print_sdk_progress(message, elapsed: float) -> None:
    """Print a short, real-time status line for a streamed SDK message, if useful."""
    try:
        from claude_agent_sdk import AssistantMessage, ToolUseBlock
    except ImportError:
        return

    if isinstance(message, AssistantMessage):
        for block in getattr(message, "content", []) or []:
            if isinstance(block, ToolUseBlock):
                tool_input = getattr(block, "input", None) or {}
                detail = tool_input.get("file_path") or tool_input.get("command") or ""
                detail = f" ({_clip(str(detail), 60)})" if detail else ""
                print(f"  [{int(elapsed)}s] Claude Code: using {block.name}{detail}", flush=True)


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
    start = time.time()
    last_heartbeat = start
    async for message in query(prompt=prompt, options=options):
        collected.extend(_collect_text_from_sdk_message(message))
        now = time.time()
        _print_sdk_progress(message, now - start)
        # Heartbeat in case a step (e.g. a long Bash call) produces no messages for a while.
        if now - last_heartbeat >= 20:
            print(f"  [{int(now - start)}s] Claude Code still working...", flush=True)
            last_heartbeat = now

    print(f"Claude Code finished in {int(time.time() - start)}s.", flush=True)
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

    start = time.time()
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )

    output_lines: list[str] = []
    deadline = start + 60 * 30  # 30 minutes
    for line in process.stdout or []:
        output_lines.append(line)
        stripped = line.strip()
        if stripped:
            print(f"  Claude Code CLI: {_clip(stripped, 160)}", flush=True)
        if time.time() > deadline:
            process.kill()
            raise RuntimeError("Claude Code CLI timed out after 30 minutes")

    returncode = process.wait()
    output = "".join(output_lines)
    print(f"Claude Code CLI finished in {int(time.time() - start)}s.", flush=True)
    if returncode != 0:
        raise RuntimeError(f"Claude Code CLI failed (exit {returncode}): {_clip(output)}")

    return output.strip() or "Claude Code CLI finished without output."


def _run_claude_code(prompt: str, cwd: str, model: str, api_key: str) -> str:
    try:
        import claude_agent_sdk  # noqa: F401
        h9.event("Claude Code", "Using claude-agent-sdk Python package")
        return asyncio.run(_run_claude_agent_sdk(prompt, cwd, model, api_key))
    except ImportError:
        print("claude-agent-sdk not installed; falling back to the 'claude' CLI...", flush=True)
        h9.event("Claude Code", "claude-agent-sdk not installed; falling back to CLI")
    except Exception as e:
        print(f"Claude Code SDK failed ({e}); falling back to the 'claude' CLI...", flush=True)
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


def _github_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


SKIP_DIR_NAMES = {".git", "__pycache__", ".venv", "node_modules", ".storage"}


def _github_json(method: str, url: str, token: str, **kwargs) -> requests.Response:
    return requests.request(
        method,
        url,
        headers=_github_headers(token),
        timeout=kwargs.pop("timeout", 60),
        **kwargs,
    )


def _repo_upstream(owner: str, name: str, token: str) -> Optional[tuple[str, str]]:
    """If owner/name is a fork, return the parent/source repo."""
    response = _github_json("GET", f"https://api.github.com/repos/{owner}/{name}", token, timeout=30)
    if response.status_code != 200:
        return None
    data = response.json() or {}
    for key in ("parent", "source"):
        full_name = (data.get(key) or {}).get("full_name")
        if full_name:
            try:
                return _parse_github_repo(full_name)
            except ValueError:
                continue
    return None


def _resolve_base_branch(owner: str, name: str, token: str) -> str:
    response = _github_json("GET", f"https://api.github.com/repos/{owner}/{name}", token, timeout=30)
    if response.status_code != 200:
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
    return default_branch


def _clone_via_github_zip(owner: str, name: str, token: str, dest: str, base_branch: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{name}/zipball/{base_branch}"
    response = requests.get(url, headers=_github_headers(token), timeout=120, allow_redirects=True)
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
    return dirs[0] if len(dirs) == 1 else dest


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
    head_owner: Optional[str] = None,
) -> tuple[str, int]:
    """
    Open a PR on owner/name.

    Same-repo: head is the branch name.
    Cross-fork: head_owner is the fork owner and head is "fork_owner:branch",
    posted against the upstream repo.
    """
    head_ref = f"{head_owner}:{head}" if head_owner and head_owner != owner else head
    query_head = f"{head_owner or owner}:{head}"
    url = f"https://api.github.com/repos/{owner}/{name}/pulls"
    response = requests.post(
        url,
        headers=_github_headers(token),
        json={"title": title, "head": head_ref, "base": base, "body": body},
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
            params={"head": query_head, "base": base, "state": "open"},
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
    pr_repo: str = "",
):
    """
    Download a GitHub repo, apply changes with Claude Code, and open a PR.

    `repo` is where the feature branch is pushed (often a fork).
    `pr_repo` is the repository that receives the pull request (often upstream).
    """
    model = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    commit_message = (commit_message or DEFAULT_COMMIT_MESSAGE).strip() or DEFAULT_COMMIT_MESSAGE
    work_branch = _normalize_work_branch(branch, prompt)
    title = (pr_title or "").strip() or commit_message

    task_start = time.time()
    print(f"Starting Claude Code GitHub task for {repo}...", flush=True)

    try:
        github_pat = _require_env("GITHUB_PAT")
        anthropic_key = _require_env("ANTHROPIC_API_KEY")
        owner, name = _parse_github_repo(repo)
        pr_owner, pr_name = _parse_github_repo(pr_repo or repo)
    except ValueError as e:
        return f"Error: {e}"

    # Models often set pr_repo equal to the fork. If this repo is a fork and
    # no different destination was given, open the PR against the upstream.
    if (pr_owner, pr_name) == (owner, name):
        upstream = _repo_upstream(owner, name, github_pat)
        if upstream:
            pr_owner, pr_name = upstream

    public_url = f"https://github.com/{owner}/{name}"
    pr_url_base = f"https://github.com/{pr_owner}/{pr_name}"
    cross_fork = (pr_owner, pr_name) != (owner, name)
    work_root: Optional[str] = None
    try:
        work_root = tempfile.mkdtemp(prefix="hal9-claude-code-")
        dest = os.path.join(work_root, name)
        fork_base = _resolve_base_branch(owner, name, github_pat)
        pr_base = _resolve_base_branch(pr_owner, pr_name, github_pat) if cross_fork else fork_base

        print(f"Downloading repository {public_url} (branch {fork_base})...", flush=True)
        h9.event("Claude Code", f"Downloading {public_url} ({fork_base})")
        repo_dir = _clone_via_github_zip(owner, name, github_pat, dest, fork_base)
        before_snapshot = _snapshot_files(repo_dir)
        print(f"Downloaded {len(before_snapshot)} files.", flush=True)

        print(f"Running Claude Code (model={model}) to apply changes...", flush=True)
        h9.event("Claude Code", f"Running Claude Code (model={model}) on branch {work_branch}")
        claude_summary = _run_claude_code(prompt, repo_dir, model, anthropic_key)

        print(f"Pushing branch '{work_branch}'...", flush=True)
        commit_sha = _publish_changes_via_github_api(
            owner=owner,
            name=name,
            token=github_pat,
            repo_dir=repo_dir,
            work_branch=work_branch,
            base_branch=fork_base,
            commit_message=commit_message,
            before=before_snapshot,
        )
        if not commit_sha:
            print("No file changes were made; skipping pull request.", flush=True)
            return (
                f"Claude Code ran on {public_url} but made no file changes, so no pull request was opened.\n\n"
                f"Claude summary:\n{claude_summary}"
            )
        print(f"Branch '{work_branch}' pushed (commit {commit_sha[:7]}).", flush=True)

        print("Opening pull request...", flush=True)
        h9.event(
            "Claude Code",
            f"Opening PR {owner}:{work_branch} -> {pr_owner}/{pr_name}:{pr_base}"
            if cross_fork else
            f"Opening PR {work_branch} -> {pr_base}",
        )
        pr_body = (
            f"{prompt.strip()}\n\n"
            f"---\n"
            f"Opened automatically by Hal9 Claude Code.\n\n"
            f"Claude summary:\n{claude_summary}"
        )
        pr_url, pr_number = _create_pull_request(
            owner=pr_owner,
            name=pr_name,
            token=github_pat,
            title=title,
            head=work_branch,
            base=pr_base,
            body=pr_body,
            head_owner=owner if cross_fork else None,
        )
        print(f"Pull request opened: {pr_url}", flush=True)
        print(f"Done in {int(time.time() - task_start)}s.", flush=True)

        return (
            f"Opened pull request #{pr_number} into {pr_url_base} ({pr_base}).\n"
            f"PR: {pr_url}\n"
            f"From: {public_url} branch {work_branch}\n"
            f"Commit: {commit_sha}\n\n"
            f"Claude summary:\n{claude_summary}"
        )
    except Exception as e:
        msg = _redact_token(str(e))
        print(f"Error: {msg}", flush=True)
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
            "Downloads a GitHub repository, uses Claude Code (ANTHROPIC_API_KEY) to make the "
            "requested code changes, pushes a feature branch, and opens a pull request using "
            "GITHUB_PAT. If the user names a fork and a source/upstream repo, set repo to the "
            "fork (where the branch is pushed) and pr_repo to the upstream (where the PR is opened). "
            "Never pushes directly to main."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": (
                        "GitHub repository to download and push the feature branch to, as "
                        "'owner/repo' or a github.com URL. For a fork workflow this is the fork "
                        "(e.g. 'hal9oo1/hal9-agent')."
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
                "pr_repo": {
                    "type": "string",
                    "description": (
                        "Repository that should receive the pull request, as 'owner/repo'. "
                        "When contributing from a fork, this MUST be the upstream/source repo "
                        "(e.g. repo='hal9oo1/hal9-agent', pr_repo='hal9ai/hal9-agent'). "
                        "Do not copy `repo` into this field if the user named a different destination."
                    ),
                },
            },
            "required": ["repo", "prompt", "branch", "model", "commit_message", "pr_title", "pr_repo"],
            "additionalProperties": False,
        },
    },
}
