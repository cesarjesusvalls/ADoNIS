"""EXACT per-event reweight from the full-record event bank: w(theta) for any knob vector, no Taylor.
Reproduces model_hist_full's per-event weight (hard-vertex amps2 x FSI kind-1 x spectral-function x norms),
using the identity-padded amps2 records so QE/RES are handled channel-correctly with no masking.
JAX-differentiable -> gradients at plot time are jax.grad(sum of bank_weight); ratios are exact at any theta.
"""
import numpy as np
import jax
import jax.numpy as jnp

from adonis.analysis.ma_records import ma_reweight, strength_reweight
from adonis.analysis.sf_reweight import sf_grids, sf_reweight, removal_from_struck
from adonis.fsi.cascade_full import pool_fsi_reweight
from adonis.xsec.spectral import SpectralFunction
from adonis.workflow.materials import resolve_targets

_FSI_F = ("bc", "sa", "ss_el", "ss", "si", "nh", "hh", "a", "iso", "finel", "inel", "swap", "ns")


def default_grids():
    return sf_grids(SpectralFunction(resolve_targets("C")[0][0].spectral_n))


def bank_weight(B, knobs, grids):
    """Exact per-event weight w(theta) (N,).  knobs: full nominal_knobs-style dict; grids: sf_grids output."""
    k = knobs
    def ma(name): return (B[f"hv_{name}_a"], B[f"hv_{name}_b"], B[f"hv_{name}_c"], B[f"hv_{name}_Q2"])
    qe_ma, res_ma = ma("qe_ma"), ma("res_ma")
    hv = (ma_reweight(qe_ma, k["M_A_qe"]) * ma_reweight(res_ma, k["M_A_res"])
          * strength_reweight(qe_ma, k["axial_strength"]) * strength_reweight(res_ma, k["res_axial_strength"])
          * strength_reweight(ma("qe_vec"), k["vector_strength"])
          * strength_reweight(ma("qe_gmp"), k["mu_p"]) * strength_reweight(ma("qe_gmn"), k["mu_n"])
          * strength_reweight(ma("qe_gep"), k["gep"]) * strength_reweight(ma("qe_gen"), k["gen"])
          * strength_reweight(ma("res_pp"), k["pion_pole"]))
    rec = {f: jnp.asarray(B[f"f_{f}"]) for f in _FSI_F}
    fsi = pool_fsi_reweight(rec, k["sabs"], 1.0, s_piN_elastic=k["s_piN_elastic"], s_piN_cex=k["s_piN_cex"],
                            s_conv=k["s_conv"], s_NN_elastic=k["s_NN_elastic"], s_NN_inelastic=k["s_NN_inelastic"],
                            f_NN_cex=k["f_NN_cex"])
    pmag, erem = removal_from_struck(jnp.asarray(B["p_struck"]))
    sfw = sf_reweight(grids, pmag, erem, kF_sf=k["kF_sf"], Eb_shift=k["Eb_shift"],
                      sf_norm=k["sf_norm"], src_tail=k["src_tail"])
    norm = jnp.where(jnp.asarray(B["channel"]) == 0, k["qe_norm"], k["res_norm"])
    return jnp.asarray(B["w0"]) * norm * hv * fsi * sfw


def to_jax(B):
    """One-time conversion of the fields bank_weight reads to on-device jnp arrays (avoids re-converting the
    ~250 MB FSI records on every reweight call)."""
    keys = [k for k in B if k.startswith("hv_") or k.startswith("f_")] + ["p_struck", "channel", "w0"]
    return {k: jnp.asarray(B[k]) for k in keys}


weight_jit = jax.jit(bank_weight)        # JB (jnp pytree) stays on device, compiled once, knobs vary
