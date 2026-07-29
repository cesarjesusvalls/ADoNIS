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
    J = np.asarray(jax.jacfwd(binned)(theta0))             # (nb, 28)
    # stat error per bin: sqrt(sum w0^2) over the selected events in the bin (nominal weights)
    w0 = np.asarray(B["w0"])[idx][keep]
    stat = np.sqrt(np.bincount(binid[keep], weights=w0 ** 2, minlength=nb))
    return J, nominal, stat


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
