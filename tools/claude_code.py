"""
Clone a GitHub repository, apply changes with Claude Code, and push back.

Uses GITHUB_PAT for git auth and ANTHROPIC_API_KEY for Claude.
The repo is cloned into a system temp directory (not ./.storage) so it is
never uploaded to Hal9 via h9.save().
"""

import asyncio
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional
from urllib.parse import urlparse

import hal9 as h9


DEFAULT_BRANCH = "main"
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


def _run_git(args: list[str], cwd: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr}")
    return result


def _clone_repo(clone_url: str, dest: str, branch: str) -> None:
    # Shallow clone of the target branch when it exists; fall back to default HEAD.
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, clone_url, dest],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return

    # Branch may not exist yet — clone default branch, then create/checkout target.
    result = subprocess.run(
        ["git", "clone", "--depth", "1", clone_url, dest],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        # Avoid leaking the token in error messages
        stderr = re.sub(r"x-access-token:[^@\s]+@", "x-access-token:***@", stderr)
        raise RuntimeError(f"Failed to clone repository: {stderr}")

    _run_git(["checkout", "-B", branch], cwd=dest)


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


def _commit_and_push(
    cwd: str,
    branch: str,
    commit_message: str,
) -> str:
    _run_git(["config", "user.email", "hal9@hal9.com"], cwd=cwd)
    _run_git(["config", "user.name", "Hal9 Claude Code"], cwd=cwd)

    _run_git(["add", "-A"], cwd=cwd)

    if not _has_changes(cwd):
        # Nothing staged after add (e.g. only ignored files)
        return "No file changes to commit."

    # Allow empty=False; commit will fail if nothing staged which we already checked
    commit = _run_git(["commit", "-m", commit_message], cwd=cwd)
    push = _run_git(["push", "-u", "origin", f"HEAD:{branch}"], cwd=cwd)

    return (
        f"Committed and pushed to '{branch}'.\n"
        f"Commit: {(commit.stdout or '').strip()}\n"
        f"Push: {(push.stdout or push.stderr or '').strip()}"
    )


def claude_code_github(
    repo: str,
    prompt: str,
    branch: str = DEFAULT_BRANCH,
    model: str = DEFAULT_MODEL,
    commit_message: str = DEFAULT_COMMIT_MESSAGE,
):
    """
    Clone a GitHub repo into a temp directory, apply changes with Claude Code,
    commit, and push to the given branch.

    Args:
        repo: GitHub repo as 'owner/repo' or full github.com URL.
        prompt: Natural-language description of the changes Claude Code should make.
        branch: Branch to push to (default: main).
        model: Claude model alias or id (default: sonnet).
        commit_message: Git commit message for the applied changes.
    """
    branch = (branch or DEFAULT_BRANCH).strip() or DEFAULT_BRANCH
    model = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    commit_message = (commit_message or DEFAULT_COMMIT_MESSAGE).strip() or DEFAULT_COMMIT_MESSAGE

    try:
        github_pat = _require_env("GITHUB_PAT")
        anthropic_key = _require_env("ANTHROPIC_API_KEY")
        owner, name = _parse_github_repo(repo)
    except ValueError as e:
        return f"Error: {e}"

    clone_url = _authenticated_clone_url(owner, name, github_pat)
    public_url = f"https://github.com/{owner}/{name}"

    work_root: Optional[str] = None
    try:
        work_root = tempfile.mkdtemp(prefix="hal9-claude-code-")
        repo_dir = os.path.join(work_root, name)

        h9.event("Claude Code", f"Cloning {public_url} (branch={branch}) into temp dir")
        _clone_repo(clone_url, repo_dir, branch)

        # Ensure remote stays authenticated for push without printing token
        _run_git(["remote", "set-url", "origin", clone_url], cwd=repo_dir)

        h9.event("Claude Code", f"Running Claude Code (model={model})")
        claude_summary = _run_claude_code(prompt, repo_dir, model, anthropic_key)

        if not _has_changes(repo_dir):
            return (
                f"Claude Code ran on {public_url} (branch '{branch}') but made no file changes.\n\n"
                f"Claude summary:\n{claude_summary}"
            )

        h9.event("Claude Code", f"Pushing changes to origin/{branch}")
        git_summary = _commit_and_push(repo_dir, branch, commit_message)

        return (
            f"Successfully updated {public_url} on branch '{branch}'.\n\n"
            f"{git_summary}\n\n"
            f"Claude summary:\n{claude_summary}"
        )
    except Exception as e:
        msg = str(e)
        msg = re.sub(r"x-access-token:[^@\s]+@", "x-access-token:***@", msg)
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
            "Clones a GitHub repository into a temporary directory (not Hal9 storage), "
            "uses Claude Code to implement the requested changes, then commits and pushes "
            "to the specified branch (default main). Requires GITHUB_PAT and ANTHROPIC_API_KEY. "
            "Use when the user wants Claude Code to modify a GitHub repo and push the result."
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
                        "Git branch to push to. Use 'main' unless the user specifies another branch."
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
            },
            "required": ["repo", "prompt", "branch", "model", "commit_message"],
            "additionalProperties": False,
        },
    },
}
