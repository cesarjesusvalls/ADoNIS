"""Gates for the flux-folded TKI pipeline building blocks: the Spectrum (histogram) flux and
the RealCascadeFSI EventRecord wrapper of the validated pion cascade."""
import numpy as np
import jax
import pytest
from pathlib import Path

jax.config.update("jax_enable_x64", True)

REL = Path(__file__).resolve().parents[1] / "_paper_release" / "AchillesGen-arXiv-2508.19213-48a9148"
_T2K = REL / "T2K" / "T2K_nu.dat"


@pytest.mark.skipif(not _T2K.exists(), reason="paper-release flux not present")
def test_spectrum_flux_t2k():
    from adonis.flux.spectrum import Spectrum
    s = Spectrum(_T2K)
    assert s.self_test()
    # T2K ND280 numu flux peaks ~0.6 GeV; flux-weighted mean ~0.8-0.9 GeV
    assert 700 < s.e_nu_nominal < 1000, s.e_nu_nominal
    e = np.asarray(s.sample_enu(jax.random.PRNGKey(1), 50_000))
    assert e.min() >= 0 and np.median(e) < s.e_nu_nominal       # right-skewed (high-E tail)


def test_real_cascade_fsi_on_eventrecord():
    """RealCascadeFSI.apply absorbs a sensible fraction of pi+ and removes (pid->0) only those,
    leaving survivors on-shell with degraded momentum."""
    import jax.numpy as jnp
    from adonis.core.event import EventRecord
    from adonis.fsi.cascade_real import RealCascadeFSI, RealCascadeConfig
    n = 4000
    key = jax.random.PRNGKey(0)
    # mono-energetic pi+ at the Delta (~300 MeV) + dummy muon/nucleon
    p = 300.0; E = np.sqrt(139.57 ** 2 + p ** 2)
    p_pi = jnp.tile(jnp.array([E, 0.0, 0.0, p]), (n, 1))
    z = jnp.zeros((n, 4))
    ev = EventRecord(k=z, kp=z, p_struck=z, p_pi=p_pi, p_N=z, w=jnp.ones(n),
                     channel=jnp.zeros(n, jnp.int32), pid_pi=jnp.full((n,), 211),
                     pid_N=jnp.full((n,), 2212), pid_Ni=jnp.full((n,), 2212),
                     W=jnp.full((n,), 1232.0), Q2_adj=jnp.ones(n))
    fsi = RealCascadeFSI(RealCascadeConfig(seed=2, step=0.08, max_steps=160))
    post = fsi.apply(None, ev, key=key)
    pid = np.asarray(post.pid_pi)
    absorbed = pid == 0
    frac = absorbed.mean()
    assert 0.02 < frac < 0.4, frac                              # some pi+ absorbed, not all
    # absorbed pions are zeroed; survivors keep a physical 4-momentum
    surv = ~absorbed
    p_out = np.asarray(post.p_pi)
    assert np.allclose(p_out[absorbed], 0.0)
    psurv = np.linalg.norm(p_out[surv][:, 1:], axis=1)
    assert np.all(psurv <= p + 1e-3) and np.median(psurv) > 0   # degraded but present


def test_nn_elastic_sigma_textbook():
    """ACHILLES GiBUU NN elastic: pp ~24 mb, np ~31 mb near p_lab ~ 1 GeV (textbook)."""
    import numpy as np
    from adonis.fsi.nucleon_cascade import nn_elastic_sigma
    M_N = 938.91875; mn = M_N / 1000.0
    def sqrts(plab):
        E = np.sqrt(plab ** 2 + mn ** 2); return np.sqrt(2 * mn ** 2 + 2 * mn * E) * 1000.0
    pp = float(nn_elastic_sigma(sqrts(1.0), True)); npn = float(nn_elastic_sigma(sqrts(1.0), False))
    assert 20 < pp < 30, pp
    assert 27 < npn < 36, npn
    # np >> pp at low energy (1/threshold), and both fall toward the ~22 mb asymptote
    assert float(nn_elastic_sigma(sqrts(0.3), False)) > 100


def test_nucleon_fsi_scatters_and_degrades():
    """A 550 MeV proton in carbon rescatters a sizeable fraction and loses energy (proton FSI)."""
    import numpy as np, jax.numpy as jnp, jax
    from adonis.fsi.nucleon_cascade import propagate_nucleon, NucleonCascadeConfig
    from adonis.fsi.cascade_real import sample_vertex
    M_N = 938.91875; n = 3000; p = 550.0; E = np.sqrt(M_N ** 2 + p ** 2)
    pN = jnp.tile(jnp.array([E, 0., 0., p]), (n, 1))
    pos0 = sample_vertex(jax.random.PRNGKey(1), n)
    pout, nsc = propagate_nucleon(pos0, pN, jnp.ones(n, bool),
                                  NucleonCascadeConfig(seed=2, step=0.08, max_steps=160),
                                  jax.random.PRNGKey(3))
    nsc = np.asarray(nsc)
    assert 0.2 < (nsc > 0).mean() < 0.7, (nsc > 0).mean()
    pmag = np.linalg.norm(np.asarray(pout)[:, 1:], axis=1)
    assert pmag.mean() < p                                # scattered protons lose energy
