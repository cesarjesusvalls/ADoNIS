"""Gate for the DCC meson-baryon cascade cross sections (adonis/fsi/mb/cascade_mb.py) -- the
piN sigma(W) per charge channel the ACHILLES Virtual-Resonances cascade scatters through
(= Phase E ANL-Osaka amplitudes).  These must be sharply Delta-peaked (the resonance shape),
carry the right charge-exchange structure, and agree with the standalone Phase-E channel
cross sections.
"""
import numpy as np
import pytest

from adonis.io import achilles_data_root
from adonis.fsi.mb import cascade_mb as cm

_ANL = achilles_data_root() / "MesonBaryonAmplitudes" / "ANL" / "ANL_0-0.dat"
pytestmark = pytest.mark.skipif(not _ANL.exists(), reason="ANL meson-baryon table not present")

M_PI = 139.57018
M_N = 938.918754


def _W_of_T(T):
    E = T + M_PI
    return np.sqrt(M_PI ** 2 + M_N ** 2 + 2 * M_N * E)


def test_dcc_scatter_delta_peaked():
    """pi+ total scatter sigma(W) is sharply peaked at the Delta (W~1232) and falls off fast
    above it (the resonance shape -- unlike the broad Oset QE)."""
    Ts = np.array([44, 121, 164, 210, 284, 330])
    tot = np.array([cm.channel_sigmas(_W_of_T(T), 0).sum() for T in Ts])
    assert np.all(tot >= 0)
    pk = Ts[int(tot.argmax())]
    assert 120 <= pk <= 200, pk                              # peaks in the Delta region
    assert tot.max() > 90, tot.max()                         # ~130 mb at the peak
    assert tot[-1] < 0.45 * tot.max()                        # falls off sharply above the Delta


def test_charge_exchange_present():
    """pi+ scatters elastically (-> pi+) AND charge-exchanges (-> pi0, on neutrons); no
    double charge flip (pi+ -> pi-)."""
    sig = cm.channel_sigmas(_W_of_T(164), 0).ravel()         # (3,) to pi+/pi0/pi-
    assert sig[0] > 0 and sig[1] > 0                          # elastic + charge exchange
    assert sig[2] == 0.0                                     # no pi+ -> pi-
    assert sig[1] / sig[0] < 0.5                              # cex sub-dominant to elastic


def test_jax_matches_numpy():
    """The JAX cascade interface reproduces the numpy channel_sigmas."""
    import jax.numpy as jnp
    W = _W_of_T(164)
    npv = cm.channel_sigmas(W, 0).ravel()
    jxv = np.asarray(cm.jax_channel_sigmas(jnp.array([W]), jnp.array([0]))[0])
    assert np.allclose(npv, jxv, rtol=1e-4), (npv, jxv)
