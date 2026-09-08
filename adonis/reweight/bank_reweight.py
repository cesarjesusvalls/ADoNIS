"""Exact per-event reweight from the full-record event bank: w(theta) for any knob vector, no Taylor.
Reproduces model_hist_full's per-event weight (hard-vertex amps2 x FSI kind-1 x spectral-function x
norms), using the identity-padded amps2 records so QE/RES are handled channel-correctly with no
masking.  JAX-differentiable, so gradients at plot time are jax.grad(sum of bank_weight); ratios are
exact at any theta.
"""
import numpy as np
import jax
import jax.numpy as jnp

from adonis.reweight.sf_reweight import sf_grids, sf_reweight, removal_from_struck
from adonis.fsi.cascade import pool_fsi_reweight
from adonis.nuclear.spectral import SpectralFunction
from adonis.nuclear.targets import resolve_targets

_FSI_F = ("bc", "sa", "ss_el", "ss", "si", "pi_hh", "pi_a", "sa_c", "ss_el_c", "ss_c", "si_c", "p_eidx",
          "hh", "a", "iso", "finel", "inel", "swap", "n_eidx")


def default_grids():
    return sf_grids(SpectralFunction(resolve_targets("C")[0][0].spectral_n))


def bank_weight(B, knobs, grids):
    """Exact per-event weight w(theta) (N,).  knobs: a PhysicsParams (adonis.core.params); grids: sf_grids output."""
    k = knobs
    if "hv_qe_mij" not in B or "hv_res_mij" not in B:
        raise KeyError(
            "bank_weight: bank carries no reduced-quadratic hard-vertex records (hv_qe_mij / "
            "hv_res_mij). A bank holding only per-knob (a,b,c) records drops the cross terms between "
            "correlated knobs and understates their degeneracy, so it is not reweightable here; "
            "regenerate it. NC banks carry no hard-vertex records at all (see the NC selection "
            "helpers in adonis/workflow).")
    from adonis.reweight.reduced_amps2 import qe_reduced_reweight, res_reduced_reweight
    probe = "EM" if "hv_qe_em" in B else "CC"
    qe = qe_reduced_reweight({"M": B["hv_qe_mij"], "Q2": B["hv_qe_Q2"]}, k,
                             probe=probe, is_proton=B.get("hv_qe_isp"))
    res = res_reduced_reweight({"M": B["hv_res_mij"], "Q2": B["hv_res_Q2"]}, k)
    hv = qe * res
    rec = {f: jnp.asarray(B[f"f_{f}"]) for f in _FSI_F}
    rec["n_events"] = len(B["w0"])
    fsi = pool_fsi_reweight(rec, k.sabs, 1.0, s_piN_elastic=k.s_piN_elastic, s_piN_cex=k.s_piN_cex,
                            s_conv=k.s_conv, s_NN_elastic=k.s_NN_elastic, s_NN_inelastic=k.s_NN_inelastic,
                            f_NN_cex=k.f_NN_cex)
    pmag, erem = removal_from_struck(jnp.asarray(B["p_struck"]))
    sfw = sf_reweight(grids, pmag, erem, kF_sf=k.kF_sf, Eb_shift=k.Eb_shift,
                      sf_norm=k.sf_norm, src_tail=k.src_tail)
    norm = jnp.where(jnp.asarray(B["channel"]) == 0, k.qe_norm, k.res_norm)
    return jnp.asarray(B["w0"]) * norm * hv * fsi * sfw


def to_jax(B):
    """One-time conversion of the fields bank_weight reads to on-device jnp arrays (avoids re-converting
    the FSI records on every reweight call).

    The EM/CC probe picks a branch rather than entering the arithmetic, so it travels as the presence
    of `hv_qe_em` and never as a value: callers pass this dict as a jitted argument, where every value
    in it is a tracer that cannot be branched on, while the key set is static pytree structure."""
    keys = [k for k in B if (k.startswith("hv_") or k.startswith("f_")) and k != "hv_qe_probe_em"]
    out = {k: jnp.asarray(B[k]) for k in keys + ["p_struck", "channel", "w0"]}
    if "hv_qe_probe_em" in B and int(np.asarray(B["hv_qe_probe_em"])):
        out["hv_qe_em"] = jnp.asarray(True)
    return out


weight_jit = jax.jit(bank_weight)
