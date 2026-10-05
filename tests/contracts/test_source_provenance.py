"""Exercise source attribution against real repositories and archive layouts."""
import os
import subprocess

import pytest

from jscc import runtime, studies


def git(root, *args):
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    return subprocess.check_output(
        ["git", *args], cwd=root, env=env, stderr=subprocess.PIPE, text=True
    ).strip()


def repository(root):
    root.mkdir()
    git(root, "init")
    (root / "tracked.txt").write_text("original\n")
    git(root, "add", ".")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-m", "fixture")
    return root


@pytest.fixture(params=[runtime, studies], ids=["runtime", "studies"])
def source_state(request, monkeypatch):
    module = request.param

    def read(root):
        monkeypatch.setattr(module, "__file__", str(root / "jscc" / "module.py"))
        return module.source_state() if module is runtime else module._source_state()

    return read


def test_checkout_tracks_clean_modified_and_untracked_state(tmp_path, source_state):
    root = repository(tmp_path / "source")
    revision = git(root, "rev-parse", "HEAD")
    assert source_state(root) == {"revision": revision, "dirty": False}
    (root / "tracked.txt").write_text("modified\n")
    assert source_state(root) == {"revision": revision, "dirty": True}
    git(root, "checkout", "--", "tracked.txt")
    (root / "untracked.txt").write_text("new\n")
    assert source_state(root) == {"revision": revision, "dirty": True}


def test_archive_inside_repository_does_not_claim_parent(tmp_path, source_state):
    parent = repository(tmp_path / "historical")
    archive = parent / "releases" / "source"
    archive.mkdir(parents=True)
    assert git(archive, "rev-parse", "HEAD") == git(parent, "rev-parse", "HEAD")
    assert source_state(archive) == {"revision": None, "dirty": None}


def test_directory_without_repository_is_unknown(tmp_path, source_state):
    assert source_state(tmp_path) == {"revision": None, "dirty": None}


def test_repository_without_commit_is_unknown(tmp_path, source_state):
    git(tmp_path, "init")
    assert source_state(tmp_path) == {"revision": None, "dirty": None}


def test_unavailable_git_is_unknown(tmp_path, monkeypatch, source_state):
    root = repository(tmp_path / "source")
    monkeypatch.setenv("PATH", "")
    assert source_state(root) == {"revision": None, "dirty": None}


def test_linked_worktree_retains_its_own_state(tmp_path, source_state):
    primary = repository(tmp_path / "primary")
    worktree = tmp_path / "linked"
    git(primary, "worktree", "add", "--detach", str(worktree))
    assert (worktree / ".git").is_file()
    revision = git(worktree, "rev-parse", "HEAD")
    assert source_state(worktree) == {"revision": revision, "dirty": False}
    (worktree / "tracked.txt").write_text("linked change\n")
    assert source_state(worktree) == {"revision": revision, "dirty": True}
    assert source_state(primary) == {"revision": revision, "dirty": False}


@pytest.mark.parametrize("redirect", ["repository", "index", "config", "discovery"])
def test_inherited_git_environment_cannot_redirect_provenance(
    tmp_path, monkeypatch, source_state, redirect
):
    root = repository(tmp_path / "source")
    other = repository(tmp_path / "other")
    revision = git(root, "rev-parse", "HEAD")
    if redirect == "repository":
        monkeypatch.setenv("GIT_DIR", str(other / ".git"))
        monkeypatch.setenv("GIT_WORK_TREE", str(other))
        monkeypatch.setenv("GIT_COMMON_DIR", str(other / ".git"))
        monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(other / ".git" / "objects"))
    elif redirect == "index":
        monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "missing-index"))
    elif redirect == "config":
        monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", "status.showUntrackedFiles")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", "no")
    else:
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert source_state(root) == {"revision": revision, "dirty": False}
    (root / "new.txt").write_text("untracked\n")
    assert source_state(root) == {"revision": revision, "dirty": True}
    archive = other / "archive"
    archive.mkdir()
    assert source_state(archive) == {"revision": None, "dirty": None}


def test_symlink_to_checkout_resolves_to_actual_root(tmp_path, source_state):
    root = repository(tmp_path / "source")
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    assert source_state(alias) == {"revision": git(root, "rev-parse", "HEAD"), "dirty": False}
