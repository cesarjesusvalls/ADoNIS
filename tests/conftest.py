"""Shared pytest fixtures: float64 everywhere, and the CI-fast flag."""
import os

import pytest
import jax

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def fast():
    """Whether to run reduced-statistics gates (set ADONIS_CI_FAST=1 in CI)."""
    return bool(os.environ.get("ADONIS_CI_FAST"))
