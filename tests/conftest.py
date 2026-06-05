"""Shared pytest fixtures: float64 everywhere, and the oracle-data path (skipped
when the npz isn't present, e.g. a checkout before the oracle is downloaded)."""
import os
from pathlib import Path

import pytest
import jax

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parent.parent
ORACLE = ROOT / "data" / "oracle" / "oracle_finalstate.npz"


@pytest.fixture(scope="session")
def oracle_path():
    if not ORACLE.exists():
        pytest.skip(f"oracle data not present at {ORACLE} "
                    "(run scripts/make_oracle_finalstate.py or fetch the release asset)")
    return str(ORACLE)


@pytest.fixture(scope="session")
def fast():
    """Whether to run reduced-statistics gates (set ADONIS_CI_FAST=1 in CI)."""
    return bool(os.environ.get("ADONIS_CI_FAST"))
