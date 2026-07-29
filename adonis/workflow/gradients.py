"""Section-2 gradient information: exact autodiff Jacobian J[bin, knob] = d(dsigma/dx_bin)/dtheta_k of the
measurable ADoNIS distributions w.r.t. the typed PhysicsParams knobs.

The engine is `reweight.bank_reweight.bank_weight` -- the exact JAX-differentiable per-event weight w(theta)
on the paper_banks schema (hv_* amps2 x f_* kind-1 FSI x SF x norms).  A knob only REWEIGHTS events (it does
not move kinematics), so the binned histogram is exactly differentiable:  d(bin_b)/dtheta = sum_{i in b}
dw_i/dtheta.  We jacfwd the per-bin summed weight over the 28-vector at nominal.

Knob order + nominal values are the SINGLE SOURCE OF TRUTH `core.params.knob_specs(nominal_knobs())` (28
plotted/fitted knobs; pw_norm + sscat excluded).  Tuple knobs (s_NN_elastic[3], s_NN_inelastic[3]) are packed
per component.
"""
import numpy as np
import jax
import jax.numpy as jnp

from adonis.core.params import nominal_knobs, knob_specs
from adonis.reweight import bank_plot as BP
from adonis.reweight import bank_reweight as BR
from adonis.reweight.bank_reweight import bank_weight
from adonis.reweight.sf_reweight import sf_grids
from adonis.nuclear.spectral import SpectralFunction
from adonis.workflow.materials import resolve_targets
from adonis.workflow import selection as SG

_SPECS = knob_specs(nominal_knobs())              # [(name, idx|None, label, nominal)] x 28
KNOB_LABELS = [s[2] for s in _SPECS]
N_KNOBS = len(_SPECS)


def _jac_via_jvp(fn, theta0, nknob):
    """Forward-mode Jacobian as `nknob` sequential JVPs (one unit tangent each) instead of jax.jacfwd, which
    materialises all tangents at once -- the loop caps peak memory for the big (~5M-event) banks."""
    cols = []
    for k in range(nknob):
        e = jnp.zeros(nknob).at[k].set(1.0)
        _, col = jax.jvp(fn, (theta0,), (e,))
        cols.append(np.asarray(col))
    return np.stack(cols, axis=1)                 # (n_out, nknob)


def pack():
    """The 28-vector of NOMINAL knob values (the point we differentiate at), in knob_specs order."""
    return jnp.asarray([float(s[3]) for s in _SPECS])


def unpack(vec):
    """length-28 vector -> PhysicsParams (start from nominal so pw_norm/sscat keep their nominal values)."""
    k = nominal_knobs()
    scal = {}; tup = {}
    for j, (name, idx, _lab, _nom) in enumerate(_SPECS):
        if idx is None:
            scal[name] = vec[j]
        else:
            tup.setdefault(name, [None, None, None])[idx] = vec[j]
    for name, comps in tup.items():
        scal[name] = tuple(comps)
    return k._replace(**scal)


_GRID_CACHE = {}


def grids_for(material):
    """sf_grids for the bank's nucleus (C or Ar); cached."""
    if material not in _GRID_CACHE:
        sfn = resolve_targets(material)[0][0].spectral_n
        _GRID_CACHE[material] = sf_grids(SpectralFunction(sfn))
    return _GRID_CACHE[material]


def bin_jacobian(bank_dir, sd, obs_key, edges, material, B=None):
    """J (n_bins, 28), nominal dsigma/dx (n_bins,), stat err (n_bins,) for one observable on a weak/EM bank.

    sd: SignalDef selecting the topology; obs_key: which STV/kinematic observable ('dpt','dalphat','pn',...);
    edges: bin edges in the observable's units; material: 'C'|'Ar' (SF grid).  Uses the SAME loaded bank for
    the selection (indices) and the differentiable weight, so event indices align."""
    B = BP.load_bank(bank_dir) if B is None else B
    sig = SG.bank_signal(bank_dir, sd, B=B)
    idx = np.asarray(sig["idx"]); obs = np.asarray(sig[obs_key], float)
    BJ = BR.to_jax(B); grids = grids_for(material)
    nb = len(edges) - 1
    binid = np.digitize(obs, edges) - 1
    keep = (binid >= 0) & (binid < nb)
    idx_k = jnp.asarray(idx[keep]); bid = jnp.asarray(binid[keep].astype(np.int32))

    def binned(theta):
        w = bank_weight(BJ, unpack(theta), grids)          # (N,) exact per-event weight w(theta)
        return jax.ops.segment_sum(w[idx_k], bid, num_segments=nb)   # (nb,) dsigma/dx_bin

    theta0 = pack()
    nominal = np.asarray(binned(theta0))
    J = _jac_via_jvp(binned, theta0, N_KNOBS)              # (nb, 28) -- memory-bounded jvp loop
    # stat error per bin: sqrt(sum w0^2) over the selected events in the bin (nominal weights)
    w0 = np.asarray(B["w0"])[idx][keep]
    stat = np.sqrt(np.bincount(binid[keep], weights=w0 ** 2, minlength=nb))
    return J, nominal, stat


_FSI_COLS = list(range(11, 22))            # knob_specs indices of the 11 FSI knobs (sabs .. f_NN_cex)


def fsi_jacobian_beam(bank_dir, nbins=30, B=None):
    """FSI-knob Jacobian for a HADRON (pion) beam bank -- the clean FSI anchor (no hard vertex / SF).
    Reweights the reacted fraction via the kind-1 pool_fsi_reweight on the f_* records:
    eff_b(theta) = Σ_{i in p-bin b} reacted_i · w_fsi_i(theta) / n_tried_b   (πR² cancels in the fractional
    response, so we return the efficiency Jacobian).  Returns J_full (nbins, 28) with only the 11 FSI columns
    populated, nominal efficiency (nbins,), and the p-bin edges."""
    import json
    from adonis.fsi.cascade import pool_fsi_reweight
    from adonis.workflow.records import derive_flags
    B = BP.load_bank(bank_dir) if B is None else B
    man = json.load(open(f"{bank_dir}/manifest.json"))
    p = np.asarray(B["beam_p"], float)
    edges = np.linspace(man["pmin"], man["pmax"], nbins + 1)
    binid = np.clip(np.digitize(p, edges) - 1, 0, nbins - 1)
    reacted = derive_flags(B)["reacted"].astype(float)
    ntry = np.bincount(binid, minlength=nbins).astype(float)
    rec = {f: jnp.asarray(B[f"f_{f}"]) for f in BR._FSI_F}; rec["n_events"] = len(p)
    react_j = jnp.asarray(reacted); bid = jnp.asarray(binid.astype(np.int32))
    ntry_j = jnp.asarray(np.where(ntry > 0, ntry, 1.0))
    fsi_specs = _SPECS[11:22]                                   # (name, idx) for the 11 FSI knobs

    def eff(theta_fsi):
        vals, tup = {}, {}
        for j, (name, idx, _l, _n) in enumerate(fsi_specs):
            if idx is None:
                vals[name] = theta_fsi[j]
            else:
                tup.setdefault(name, [None, None, None])[idx] = theta_fsi[j]
        wf = pool_fsi_reweight(rec, vals["sabs"], 1.0, s_piN_elastic=vals["s_piN_elastic"],
                               s_piN_cex=vals["s_piN_cex"], s_conv=vals["s_conv"],
                               s_NN_elastic=tuple(tup["s_NN_elastic"]),
                               s_NN_inelastic=tuple(tup["s_NN_inelastic"]), f_NN_cex=vals["f_NN_cex"])
        return jax.ops.segment_sum(react_j * wf, bid, num_segments=nbins) / ntry_j

    theta0 = jnp.asarray([float(s[3]) for s in fsi_specs])
    nominal = np.asarray(eff(theta0))
    Jfsi = _jac_via_jvp(eff, theta0, len(fsi_specs))           # (nbins, 11) -- memory-bounded jvp loop
    Jfull = np.zeros((nbins, N_KNOBS)); Jfull[:, _FSI_COLS] = Jfsi
    return Jfull, nominal, edges


def finite_diff_check(bank_dir, sd, obs_key, edges, material, knob_index, eps=1e-3, B=None):
    """Central finite-difference of the binned cross section w.r.t. one knob, to validate jacfwd."""
    B = BP.load_bank(bank_dir) if B is None else B
    sig = SG.bank_signal(bank_dir, sd, B=B)
    idx = np.asarray(sig["idx"]); obs = np.asarray(sig[obs_key], float)
    BJ = BR.to_jax(B); grids = grids_for(material)
    nb = len(edges) - 1
    binid = np.digitize(obs, edges) - 1; keep = (binid >= 0) & (binid < nb)
    idx_k = jnp.asarray(idx[keep]); bid = jnp.asarray(binid[keep].astype(np.int32))

    def binned_vec(vec):
        w = bank_weight(BJ, unpack(jnp.asarray(vec)), grids)
        return np.asarray(jax.ops.segment_sum(w[idx_k], bid, num_segments=nb))

    v0 = np.asarray(pack()); vp = v0.copy(); vm = v0.copy()
    vp[knob_index] += eps; vm[knob_index] -= eps
    return (binned_vec(vp) - binned_vec(vm)) / (2 * eps)   # (nb,) dSigma/dtheta_knob
