"""The detector and the two grids: the parts of the unfolding chain that must be right before any fit.

These run without a bank -- they pin the contracts, not the numbers.  What they guard:

  * the smearing is a function of (seed, chunk) and NOTHING else, so the detector cannot depend on how a
    job was sharded or on how many chunks a process happened to see;
  * every particle stays on shell, including exactly-forward ones, which a naive rotation frame leaves
    unsmeared -- forward muons are most of this sample;
  * the flat bin index means the same thing everywhere.  A truth index that means one thing in the
    response builder and another in the figure produces a plausible, wrong unfolded spectrum, and nothing
    downstream would flag it.
"""
import numpy as np
import pytest

from adonis.detector import SmearSpec, smear_chunk
from adonis.unfold.binning import Grid2D, reco_grid, truth_grid


def _bank(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 400, (n, 3))
    m = 105.658
    E = np.sqrt(m ** 2 + (p ** 2).sum(1))
    p4 = np.column_stack([E, p])
    return {"k_lep": p4, "fs_p4": p4.copy(), "fs_pid": np.full(n, 2212), "w0": np.ones(n)}


# ---- detector --------------------------------------------------------------------------------------- #

def test_smearing_is_a_function_of_seed_and_chunk_only():
    B, spec = _bank(), SmearSpec()
    assert np.array_equal(smear_chunk(B, spec, 3)["k_lep"], smear_chunk(B, spec, 3)["k_lep"])
    assert not np.array_equal(smear_chunk(B, spec, 3)["k_lep"], smear_chunk(B, spec, 4)["k_lep"])
    other = SmearSpec(seed=spec.seed + 1)
    assert not np.array_equal(smear_chunk(B, other, 3)["k_lep"], smear_chunk(B, spec, 3)["k_lep"])


def test_resolutions_are_what_the_spec_says():
    B = _bank(200_000)
    R = smear_chunk(B, SmearSpec(sigma_p=0.20, sigma_theta_deg=10.0), 0)
    pt = np.linalg.norm(B["k_lep"][:, 1:], axis=1)
    pr = np.linalg.norm(R["k_lep"][:, 1:], axis=1)
    assert abs(((pr - pt) / pt).std() - 0.20) < 0.005
    u = B["k_lep"][:, 1:] / pt[:, None]
    v = R["k_lep"][:, 1:] / pr[:, None]
    ang = np.degrees(np.arccos(np.clip((u * v).sum(1), -1, 1)))
    # E|N(0,s)| = s*sqrt(2/pi): the spec quotes the 1-sigma width, not the mean deflection
    assert abs(ang.mean() - 10.0 * np.sqrt(2 / np.pi)) < 0.15


def test_particles_stay_on_shell():
    B = _bank(50_000)
    R = smear_chunk(B, SmearSpec(), 0)
    m2 = lambda p4: p4[:, 0] ** 2 - (p4[:, 1:] ** 2).sum(1)
    assert np.allclose(m2(R["k_lep"]), m2(B["k_lep"]), rtol=1e-9)


def test_exactly_forward_particles_are_smeared():
    """The rotation frame is built against the LEAST-aligned axis for this reason: crossing with z leaves
    a z-directed particle with a zero-length basis vector, and forward muons dominate this sample."""
    fwd = np.array([[1000.0, 0.0, 0.0, np.sqrt(1000.0 ** 2 - 105.658 ** 2)]])
    B = {"k_lep": np.repeat(fwd, 4000, 0), "fs_p4": np.zeros((0, 4))}
    R = smear_chunk(B, SmearSpec(), 0)
    p = R["k_lep"][:, 1:]
    ang = np.degrees(np.arccos(np.clip(p[:, 2] / np.linalg.norm(p, axis=1), -1, 1)))
    assert 6.0 < ang.mean() < 10.0, f"forward particles barely deflected: {ang.mean():.2f} deg"


def test_zero_momentum_particles_pass_through_without_nan():
    B = {"k_lep": np.zeros((5, 4)), "fs_p4": np.zeros((5, 4))}
    R = smear_chunk(B, SmearSpec(), 0)
    assert np.isfinite(R["k_lep"]).all() and np.isfinite(R["fs_p4"]).all()


def test_untouched_arrays_are_shared_not_copied():
    B = _bank()
    R = smear_chunk(B, SmearSpec(), 0)
    assert R["fs_pid"] is B["fs_pid"] and R["w0"] is B["w0"]
    assert R["k_lep"] is not B["k_lep"]


# ---- grids ------------------------------------------------------------------------------------------ #

def test_flat_index_is_dpt_slowest():
    g = Grid2D([0, 10, 20], [0, 1, 2, 3])           # 2 x 3
    assert g.n == 6
    assert g.index([5], [0.5])[0] == 0               # (dpt 0, dat 0)
    assert g.index([5], [2.5])[0] == 2               # (dpt 0, dat 2)
    assert g.index([15], [0.5])[0] == 3              # (dpt 1, dat 0)


def test_out_of_range_is_minus_one_never_clipped():
    """An event at 3 GeV is not a 1 GeV event.  Clipping it into the edge bin would put unmeasured
    events into a template that then claims to describe them."""
    g = Grid2D([0, 10, 20], [0, 1, 2])
    assert g.index([25.0], [0.5])[0] == -1
    assert g.index([-1.0], [0.5])[0] == -1
    assert g.index([5.0], [7.0])[0] == -1


def test_projection_matches_the_flattening():
    g = Grid2D([0, 10, 20], [0, 1, 2, 3])
    v = np.arange(6.0)                               # rows = dpt, cols = dat
    assert list(g.project(v, "dpt")) == [3.0, 12.0]   # 0+1+2, 3+4+5
    assert list(g.project(v, "dat")) == [3.0, 5.0, 7.0]


def test_the_shipped_grids_have_the_agreed_shape():
    T, R = truth_grid(), reco_grid()
    assert (T.ndpt, T.ndat, T.n) == (5, 2, 10)
    assert (R.ndpt, R.ndat, R.n) == (10, 3, 30)
    assert R.n > T.n, "unfolding needs more reco bins than truth bins"


@pytest.mark.parametrize("g", [truth_grid(), reco_grid()])
def test_the_dpt_axis_is_open_at_the_top(g):
    """Smearing pushes the reco tail past 4 GeV; a closed axis silently drops those events from the fit."""
    assert g.dpt[-1] == np.inf
    assert g.index([9e9], [1.0])[0] >= 0
