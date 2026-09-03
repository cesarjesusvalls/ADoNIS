"""The Python files that are part of the repository, for the tests that sweep all of them.

Walking the checkout picks up whatever else is lying in it -- a `build/` left by `python -m build`,
a virtualenv, an editable install's artifacts -- and those carry stale copies of the package, so the
sweep silently changes size and starts checking code that is not the source.  Ask git what it
tracks; fall back to a plain walk when there is no git (an unpacked sdist).
"""
from __future__ import annotations

import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
_SKIP_DIRS = ("output/", "build/", "dist/", ".venv/", "venv/", ".git/")


def _walk():
    return sorted(p for p in ROOT.rglob("*.py")
                  if "__pycache__" not in str(p)
                  and not str(p.relative_to(ROOT)).startswith(_SKIP_DIRS))


def python_files():
    """Tracked .py files, absolute, sorted.  Never includes build output or a virtualenv."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.py"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            files = sorted((ROOT / line).resolve() for line in out.stdout.splitlines() if line.strip())
            return [p for p in files if p.exists()]
    except (OSError, subprocess.SubprocessError):
        pass
    return _walk()
