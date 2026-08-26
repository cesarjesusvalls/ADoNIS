"""One sample layer: the Gate-I Jacobian and the sec4 engine must describe the same stack.

The regression this guards is concrete.  The list of samples used to live in two places -- SAMPLES in
the Gate-I builder and a literal list of _bank() calls in the sec4 engine -- and the sec4 engine
asserts its dskeys against the npz gate1 produced.  Adding the MINERvA CC1pi+ samples meant editing both
in step; getting it wrong surfaces as an AssertionError several minutes into a GPU job, or (worse) as a
Jacobian and a fit that quietly describe different data.

These tests run without banks: they compare declared structure, not computed numbers.
"""
import pathlib

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

from adonis.fit.config import FitConfig

FIT_CFG = "configs/fits/sec4_P1.yaml"
BEAMS_CFG = "configs/samples/beams.yaml"
GATE_NPZ = pathlib.Path("output/altgen/multisample_carbon.npz")


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
    src = pathlib.Path("analysis/campaign/gate1.py").read_text()
    assert "BEAM_OBS = {" not in src, "BEAM_OBS is declared in code again; it belongs in beams.yaml"
    assert "SAMPLES = [" not in src, "the sample list is hardcoded again; it belongs in the fit config"


@pytest.mark.skipif(not GATE_NPZ.exists(), reason="needs output/altgen/multisample_carbon.npz")
def test_gate_jacobian_matches_the_fit_config(cfg, beams):
    """THE GUARDRAIL.  The committed Jacobian's dskeys must be exactly what the fit config declares:
    every observable of every sample, in config order, then the beam blocks, in config order.

    This is the check the sec4 engine performs at runtime after minutes of bank loading; doing it here
    from declarations alone makes a mismatch a one-second test failure instead."""
    z = np.load(GATE_NPZ, allow_pickle=True)
    got = [str(x) for x in z["dskeys"]]

    want = []
    for nm in cfg.samples:
        s = yaml.safe_load(pathlib.Path(f"configs/samples/{nm}.yaml").read_text())
        # `fit: false` marks an observable that is PLOTTED but does not enter the Fisher/fit -- the
        # validation extras (t2k_cc1pi_ch ppi/cos_pi, the minerva_stv lepton/proton kinematics).  Only
        # the fitted ones appear in the Jacobian, so only they belong in this comparison.
        want += [f"{s['name']}:{o['key']}" for o in s["observables"] if o.get("fit", True)]
    for b in cfg.beams:
        want += list(beams["beams"][b]["observables"])

    assert got == want, (
        "the Gate-I Jacobian and the fit config describe different stacks\n"
        f"  npz    {got}\n  config {want}")


@pytest.mark.skipif(not GATE_NPZ.exists(), reason="needs output/altgen/multisample_carbon.npz")
def test_beam_blocks_are_the_declared_width(cfg, beams):
    z = np.load(GATE_NPZ, allow_pickle=True)
    row0 = [int(x) for x in z["row0"]]
    nb = int(beams["nbins"])
    widths = np.diff(row0)
    n_beam_obs = 2 * len(cfg.beams)
    assert list(widths[-n_beam_obs:]) == [nb] * n_beam_obs, \
        f"beam blocks are not {nb} bins wide; beams.yaml and the cached beam Jacobian disagree"
