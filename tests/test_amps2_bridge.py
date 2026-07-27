"""Stage-3 unification proof: the numpy generator's absolute amps2 (dcc_current.exclusive_amps2_batch)
and the differentiable jnp path's L.W (channel.weight_from_sample) compute the SAME hadron current on
identical kinematics, related by the FIRST-PRINCIPLES (zero-fit) bridge

    amps2_gen  =  LW_diff * (ee/(sw*sqrt2))^2 * 1/4 * |prop_W|^2 / _NORM

with prop_W = i/(q^2 - M_W^2 - i M_W Gamma_W).  The (ee/sw/sqrt2)*1/2 factor is the CC leptonic
vertex that the generator folds into lepton_current but the differentiable lepton_tensor_cc omits;
_NORM carries the hadronic FResV^2 (2 m_N)^2 / 2pi.  This is what licenses deleting the FITTED
SIGMA_UNIT_NB: the amplitude+coupling part of the absolute scale is derived, not tuned.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import adonis.channels.dcc.current as dcc
dcc.BATCH_INTERP = "spline"
from adonis.channels.dcc.current import exclusive_amps2_batch, _NORM
from adonis.channels import constants as C
from adonis.channels.dcc.channel import sample_final_state, weight_from_sample
from adonis.channels.dcc.structure import HadronStructure
from adonis.core.params import DCCKnobs
from adonis.nuclear.free import FreeNucleon


def test_amps2_generator_equals_diffpath_LW_bridge():
    hs = HadronStructure()                                  # CC channels (ch2 = p->p pi+)
    S = sample_final_state(jax.random.PRNGKey(0), 4000, hs=hs, e_nu=1000.0,
                           ep_lo=105.7, ep_hi=1000.0, nuclear=FreeNucleon(), m_lep=105.7)
    _, LWc = weight_from_sample(DCCKnobs(), S, use_spline=True)
    LWc = np.asarray(LWc); cut = np.asarray(S["cut"])
    mom = lambda k: np.asarray(S[k], float)
    a2 = np.asarray(exclusive_amps2_batch(mom("k_lab"), mom("kp_lab"), mom("p_struck"),
                                          mom("p_N"), mom("p_pi"), +1, 211))
    g = cut & (a2 > 0) & (LWc[:, 2] > 0) & np.isfinite(a2) & np.isfinite(LWc[:, 2])
    q = mom("k_lab") - mom("kp_lab"); q2 = q[:, 0] ** 2 - np.sum(q[:, 1:] ** 2, axis=1)
    prop2 = 1.0 / ((q2 - C.MW ** 2) ** 2 + (C.MW * C.GAMW) ** 2)
    bridge = (C.ee / (C.sw * np.sqrt(2.0))) ** 2 * 0.25 * prop2 / _NORM
    r = a2[g] / (LWc[g, 2] * bridge[g])
    assert abs(r.mean() - 1.0) < 1e-3 and r.std() < 1e-3, \
        f"bridge mismatch: mean {r.mean():.6f} std {r.std():.2e} over {int(g.sum())} events"


if __name__ == "__main__":
    test_amps2_generator_equals_diffpath_LW_bridge()
    print("OK: amps2_gen == LW_diff * derived bridge (ratio 1.000, zero-fit)")
