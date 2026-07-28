"""Shared pytest fixtures: float64 everywhere, and the CI-fast flag."""
import os

import pytest
import jax

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def fast():
    """Whether to run reduced-statistics gates (set ADONIS_CI_FAST=1 in CI)."""
    return bool(os.environ.get("ADONIS_CI_FAST"))


# --- heavy import-time test files: SKIP COLLECTING them by default -------------------------------- #
# Their module-level bank/replica/DCC-amplitude builds run at IMPORT (collection), so they dominate
# the suite wall-time even when deselected by a marker -- skipping collection is the only real speedup.
# The default run keeps the fast unit tests + the fast analogs (e.g. test_full_knobs_smoke); the full
# high-statistics gates run in CI / after many changes with `pytest --runslow`.
_HEAVY = {
    "test_full_knobs_grad.py",       # 5k-event replica + 14 DCC pw records + SF (analog: test_full_knobs_smoke)
    "test_fold_fs.py", "test_grad_fs.py",                 # DCC fold_final_state at import
    "test_ma_reweight.py", "test_res_strength_reweight.py",
    "test_sf_reweight.py", "test_strength_reweight.py",   # module-level sample_importance + amps2 records
    "test_amps2_bridge.py", "test_build_zmtx_equiv.py",   # DCC amplitude builds
    "test_generate_bank_unify.py",                        # builds weak+EM+hadron banks (QE+RES cascade) at import
}


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also collect+run the heavy import-time test files (default: skipped)")


def pytest_ignore_collect(collection_path, config):
    """Skip collecting the heavy files unless --runslow (avoids their import-time cost)."""
    if config.getoption("--runslow"):
        return False
    return collection_path.name in _HEAVY
