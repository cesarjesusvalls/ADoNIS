"""One sample layer: the constrained-set Jacobian and the sec4 engine must describe the same stack.

The sample list exists in two places -- SAMPLES in the builder, and the _bank() calls in the engine --
and the engine asserts its dskeys against the npz the builder produced.  They must be edited in step:
a mismatch surfaces minutes into a GPU job, or, worse, as a Jacobian and a fit that describe
different data.

These tests run without banks: they compare declared structure, not computed numbers.
"""
import pathlib

import numpy as np
import pytest

from analysis._cli import results_dir

yaml = pytest.importorskip("yaml")

from adonis.fit.config import FitConfig

FIT_CFG = "configs/fits/sec4_P1.yaml"
BEAMS_CFG = "configs/samples/beams.yaml"
JACOBIAN_NPZ = results_dir() / "multisample_carbon.npz"


@pytest.fixture(scope="module")
def cfg():
    return FitConfig.load(FIT_CFG)


@pytest.fixture(scope="module")
def beams():
    return yaml.safe_load(pathlib.Path(BEAMS_CFG).read_text())


def test_every_named_sample_has_a_config(cfg):
    for nm in cfg.samples:
        assert pathlib.Path(f"configs/samples/{nm}.yaml").exists(), f"no config for sample {nm!r}"


def test_beams_declared_once_and_only_once(cfg, beams):
    """BEAM_OBS was a dict hardcoded in two modules.  It now lives in configs/samples/beams.yaml, and
    nothing may redeclare it."""
    assert set(cfg.beams) <= set(beams["beams"]), "fit config names a beam beams.yaml does not declare"
    for b in cfg.beams:
        assert len(beams["beams"][b]["observables"]) == 2, \
            "beam_jacobian returns exactly two observables per beam (reaction, then the second)"
    src = pathlib.Path("analysis/campaign/constrained.py").read_text()
    assert "BEAM_OBS = {" not in src, "BEAM_OBS is declared in code again; it belongs in beams.yaml"
    assert "SAMPLES = [" not in src, "the sample list is hardcoded again; it belongs in the fit config"


@pytest.mark.skipif(not JACOBIAN_NPZ.exists(), reason="needs the constrained-set npz")
def test_gate_jacobian_matches_the_fit_config(cfg, beams):
    """THE GUARDRAIL.  The committed Jacobian's dskeys must be exactly what the fit config declares:
    every observable of every sample, in config order, then the beam blocks, in config order.

    This is the check the sec4 engine performs at runtime after minutes of bank loading; doing it here
    from declarations alone makes a mismatch a one-second test failure instead."""
    z = np.load(JACOBIAN_NPZ, allow_pickle=True)
    got = [str(x) for x in z["dskeys"]]

    want = []
    for nm in cfg.samples:
        s = yaml.safe_load(pathlib.Path(f"configs/samples/{nm}.yaml").read_text())
        want += [f"{s['name']}:{o['key']}" for o in s["observables"] if o.get("fit", True)]
    for b in cfg.beams:
        want += list(beams["beams"][b]["observables"])

    assert got == want, (
        "the constrained-set Jacobian and the fit config describe different stacks\n"
        f"  npz    {got}\n  config {want}")


@pytest.mark.skipif(not JACOBIAN_NPZ.exists(), reason="needs the constrained-set npz")
def test_beam_blocks_are_the_declared_width(cfg, beams):
    z = np.load(JACOBIAN_NPZ, allow_pickle=True)
    row0 = [int(x) for x in z["row0"]]
    nb = int(beams["nbins"])
    widths = np.diff(row0)
    n_beam_obs = 2 * len(cfg.beams)
    assert list(widths[-n_beam_obs:]) == [nb] * n_beam_obs, \
        f"beam blocks are not {nb} bins wide; beams.yaml and the cached beam Jacobian disagree"
