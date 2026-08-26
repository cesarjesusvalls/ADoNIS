"""EXACT per-event reweight from the full-record event bank: w(theta) for any knob vector, no Taylor.
Reproduces model_hist_full's per-event weight (hard-vertex amps2 x FSI kind-1 x spectral-function x norms),
using the identity-padded amps2 records so QE/RES are handled channel-correctly with no masking.
JAX-differentiable -> gradients at plot time are jax.grad(sum of bank_weight); ratios are exact at any theta.
"""
import numpy as np
import jax
import jax.numpy as jnp

from adonis.reweight.amps2_records import ma_reweight, strength_reweight
from adonis.reweight.sf_reweight import sf_grids, sf_reweight, removal_from_struck
from adonis.fsi.cascade import pool_fsi_reweight
from adonis.nuclear.spectral import SpectralFunction
from adonis.nuclear.targets import resolve_targets

# RAGGED kind-1 FSI record (see cascade.compact_fsi_record): flat per-slot arrays + a per-slot event
# index.  The dense (n, K) layout was ~97% padding; this is ~40x fewer slots to store AND to reweight.
_FSI_F = ("bc", "sa", "ss_el", "ss", "si", "pi_hh", "pi_a", "sa_c", "ss_el_c", "ss_c", "si_c", "p_eidx",
          "hh", "a", "iso", "finel", "inel", "swap", "n_eidx")


def default_grids():
    return sf_grids(SpectralFunction(resolve_targets("C")[0][0].spectral_n))


def bank_weight(B, knobs, grids):
    """Exact per-event weight w(theta) (N,).  knobs: a PhysicsParams (adonis.core.params); grids: sf_grids output."""
    k = knobs
    if "hv_qe_ma_a" not in B and "hv_qe_mij" not in B:
        # Fail loud, not with a cryptic deep KeyError.  NC RES-only banks (channels=[res]) never write
        # hard-vertex records, and NC QE nuclear generation is not implemented, so their differentiable
        # weight is undefined -- the sec2/sec3 gradient machinery does not yet support NC.  NC selections
        # go through bank_signal_nc / oracle_signal_nc, which read w0 directly and never call this.
        raise KeyError("bank_weight: bank carries no hard-vertex (hv_*) records; NC banks are not "
                       "reweightable through this path (see adonis/workflow NC selection helpers).")
    def ma(name): return (B[f"hv_{name}_a"], B[f"hv_{name}_b"], B[f"hv_{name}_c"], B[f"hv_{name}_Q2"])
    # QE and RES each take the exact reduced-quadratic M when the bank has been refreshed (hv_qe_mij /
    # hv_res_mij), else the legacy per-knob product.
    if "hv_qe_mij" in B:
        from adonis.reweight.reduced_amps2 import qe_reduced_reweight
        probe = "EM" if int(np.asarray(B.get("qe_probe_em", 0)).item() if "qe_probe_em" in B else 0) else "CC"
        qe = qe_reduced_reweight({"M": B["hv_qe_mij"], "Q2": B["hv_qe_Q2"]}, k,
                                 probe=probe, is_proton=B.get("hv_qe_isp"))
    else:
        qe_ma = ma("qe_ma")
        qe = (ma_reweight(qe_ma, k.M_A_qe) * strength_reweight(qe_ma, k.axial_strength)
              * strength_reweight(ma("qe_vec"), k.vector_strength)
              * strength_reweight(ma("qe_gmp"), k.mu_p) * strength_reweight(ma("qe_gmn"), k.mu_n)
              * strength_reweight(ma("qe_gep"), k.gep) * strength_reweight(ma("qe_gen"), k.gen))
    if "hv_res_mij" in B:
        from adonis.reweight.reduced_amps2 import res_reduced_reweight
        res = res_reduced_reweight({"M": B["hv_res_mij"], "Q2": B["hv_res_Q2"]}, k)
    else:
        res_ma = ma("res_ma")
        res = (ma_reweight(res_ma, k.M_A_res) * strength_reweight(res_ma, k.res_axial_strength)
               * strength_reweight(ma("res_pp"), k.pion_pole))
        if "hv_res_delta_a" in B:                  # optional P33 Delta-strength knob (banks that carry it)
            res = res * strength_reweight(ma("res_delta"), k.delta_strength)
    hv = qe * res
    rec = {f: jnp.asarray(B[f"f_{f}"]) for f in _FSI_F}
    rec["n_events"] = len(B["w0"])               # ragged reduction needs the event count
    fsi = pool_fsi_reweight(rec, k.sabs, 1.0, s_piN_elastic=k.s_piN_elastic, s_piN_cex=k.s_piN_cex,
                            s_conv=k.s_conv, s_NN_elastic=k.s_NN_elastic, s_NN_inelastic=k.s_NN_inelastic,
                            f_NN_cex=k.f_NN_cex)
    pmag, erem = removal_from_struck(jnp.asarray(B["p_struck"]))
    sfw = sf_reweight(grids, pmag, erem, kF_sf=k.kF_sf, Eb_shift=k.Eb_shift,
                      sf_norm=k.sf_norm, src_tail=k.src_tail)
    norm = jnp.where(jnp.asarray(B["channel"]) == 0, k.qe_norm, k.res_norm)
    return jnp.asarray(B["w0"]) * norm * hv * fsi * sfw


def to_jax(B):
    """One-time conversion of the fields bank_weight reads to on-device jnp arrays (avoids re-converting the
    ~250 MB FSI records on every reweight call)."""
    keys = [k for k in B if k.startswith("hv_") or k.startswith("f_")] + ["p_struck", "channel", "w0"]
    return {k: jnp.asarray(B[k]) for k in keys}


weight_jit = jax.jit(bank_weight)        # JB (jnp pytree) stays on device, compiled once, knobs vary
