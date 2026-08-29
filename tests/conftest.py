"""Shared pytest fixtures: float64 everywhere, and the CI-fast flag."""
import os

import pytest
import jax

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def fast():
    """Whether to run reduced-statistics gates (set ADONIS_CI_FAST=1 in CI)."""
    return bool(os.environ.get("ADONIS_CI_FAST"))


_HEAVY = {
    "test_full_knobs_grad.py",
    "test_fold_fs.py", "test_grad_fs.py",
    "test_ma_reweight.py", "test_res_strength_reweight.py",
    "test_sf_reweight.py", "test_strength_reweight.py",
    "test_amps2_bridge.py", "test_build_zmtx_equiv.py",
    "test_generate_bank_unify.py",
}


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also collect+run the heavy import-time test files (default: skipped)")


def pytest_ignore_collect(collection_path, config):
    """Skip collecting the heavy files unless --runslow (avoids their import-time cost).

    Returns True to ignore, or None to DEFER.  Never False: this hook is `firstresult`, so an
    explicit False means "definitely collect this" and VETOES pytest's own --ignore / --ignore-glob /
    --deselect.  Returning False on the common path is why `pytest tests/ --ignore=tests/foo.py`
    silently collected foo.py anyway.
    """
    if config.getoption("--runslow"):
        return None
    return True if collection_path.name in _HEAVY else None

def pytest_collection_modifyitems(config, items):
    """Skip tests marked `slow` unless --runslow.

    pytest.ini declares the marker and two files apply it, but nothing read it, so those tests ran on
    every invocation -- about 5.5 minutes of MC integration in a suite that is otherwise a few minutes.
    The `_HEAVY` list above solves a different problem: files whose IMPORT is expensive have to be
    dropped at collection, which a marker cannot do because the marker is only visible after import.
    """
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="marked slow; run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
