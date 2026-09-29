"""Best-effort Git provenance for the actual source directory."""
import os
from pathlib import Path
import subprocess


def git_source_state(root: Path):
    """Only attribute Git state when the source root is the repository root."""
    unknown = {"revision": None, "dirty": None}
    # Repository/index redirects and injected Git configuration belong to the
    # invoking process, not necessarily this source tree. Use one isolated
    # environment for discovery, revision and status (including .git worktrees).
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}

    def git(*args: str):
        return subprocess.check_output(
            ["git", *args], cwd=root, env=env, stderr=subprocess.DEVNULL, text=True
        ).strip()

    try:
        root = root.resolve()
        if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
            return unknown
        revision = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return unknown
