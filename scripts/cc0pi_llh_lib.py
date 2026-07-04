"""Shared machinery for the T2K CC0pi LLH-surface study (Callum_B / llh-surface).

The EXACT differentiable chi^2(theta) over the full T2K STV covariance, evaluated by re-summing the
frozen 1M event bank (bank_reweight.weight_jit) restricted to CC0pi-signal events.  Because the bin
index + acceptance mask are theta-independent (frozen kinematics), the model histogram is just a
segment_sum of the exact per-event weight w(theta) -> chi^2(theta) is jit + grad + hessian-able in
every knob, with NO Taylor / NO separability approximation.

Design (mirrors the CC0pi blueprint):
  * absolute model (no fitted normalization); shape-profiled (closed-form global A) is a SECONDARY view.
  * robust covariance inverse: eigen-floor (relative), condition number reported.  The STV 8x8 covs are
    well-conditioned; the eigen-floor is the same robust handling we needed on the 2D pcos cov.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root
import numpy as np
import uproot
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from analysis.t2k.differentiability import bank_plot as BP, bank_reweight as BR
from analysis.t2k.differentiability.full_knobs import nominal_knobs

BANKDIR = os.environ.get("ADONIS_EVENT_BANK", "output/event_bank")

# ---- T2K CC0pi STV data + covariance (mirror tune.py / cc0pi_data_chi2.py) --------------------- #
def load_t2k_cc0pi(obs):
    """obs in {dpt,dat}.  Returns edges [MeV|rad], conv (nb/unit/12C -> 1e-38 cm^2/unit/nuc), data, derr, cov
    (all x1e38)."""
    r = uproot.open(f"../nuisance/data/T2K/CC0pi/STV/{obs}Results.root")
    if obs == "dpt":
        edges = np.asarray(r["Result"].axis().edges()) * 1000.0
        conv = 1e-33 / 12.0 * 1000.0 * 1e38
    else:
        edges = np.asarray(r["Result"].axis().edges())
        conv = 1e-33 / 12.0 * 1e38
    data = np.asarray(r["Result"].values()) * 1e38
    derr = np.asarray(r["Result"].errors()) * 1e38
    cov = np.asarray(r["Covariance_Matrix"].values())
    return edges, conv, data, derr, cov


def robust_cinv(cov, rtol=1e-10):
    """Eigen-floored inverse of a symmetric covariance.  Floors eigenvalues at rtol*max(eig) (an SVD/pinv
    with a relative cutoff) so an ill-conditioned block cannot blow up chi^2.  Returns (Cinv, cond, nfloored)."""
    cov = 0.5 * (cov + cov.T)
    w, V = np.linalg.eigh(cov)
    wmax = float(w.max())
    floor = rtol * wmax
    nfloored = int(np.sum(w < floor))
    w_f = np.where(w < floor, floor, w)
    cinv = (V * (1.0 / w_f)) @ V.T
    cond = wmax / float(w.min()) if w.min() > 0 else np.inf
    return 0.5 * (cinv + cinv.T), cond, nfloored


# ---- bank -> CC0pi-signal sliced device pytree + per-obs bin indices --------------------------- #
def load_signal_bank(bankdir=None, verbose=True):
    """Load the 1M bank, slice to CC0pi-signal events, return (JBs, grids, aux) where JBs is the on-device
    pytree for weight_jit over signal events only, and aux carries the leading proton + full B (for obs)."""
    bankdir = bankdir or BANKDIR
    B = BP.load_bank(bankdir)
    mask, lead = BP.signal_cc0pi(B)                     # prim-pi absorbed + T2K acceptance
    mask = np.asarray(mask); nsig = int(mask.sum())
    if verbose:
        print(f"[lib] bank {bankdir}: {len(B['w0'])} events -> {nsig} CC0pi signal "
              f"({(B['channel'][mask] == 0).sum()} QE + {(B['channel'][mask] == 1).sum()} RES)", flush=True)
    JB = BR.to_jax(B)
    JBs = {k: v[jnp.asarray(mask)] for k, v in JB.items()}
    grids = BR.default_grids()
    aux = dict(B=B, mask=mask, lead=lead, nsig=nsig)
    return JBs, grids, aux


def obs_binning(aux, obs):
    """Per-signal-event bin index + (edges, conv, data, derr, cov) for a CC0pi STV observable."""
    B, mask, lead = aux["B"], aux["mask"], aux["lead"]
    edges, conv, data, derr, cov = load_t2k_cc0pi(obs)
    val = BP.dpt(B, lead) if obs == "dpt" else BP.dat(B, lead)
    val = np.asarray(val)[mask]
    nb = len(edges) - 1
    idx = np.clip(np.searchsorted(edges, val) - 1, 0, nb - 1)
    bw = np.diff(edges)
    return dict(obs=obs, idx=jnp.asarray(idx), nb=nb, bw=jnp.asarray(bw), conv=conv,
                edges=edges, data=jnp.asarray(data), derr=derr, cov=cov)


# ---- the differentiable chi^2 over a set of CC0pi observables ---------------------------------- #
def make_chi2(JBs, grids, obs_specs, rtol=1e-10):
    """Build chi2_abs(knobs), chi2_shape(knobs) (global-A profiled), and model_vec(knobs) over a LIST of obs
    specs (each from obs_binning).  Uses one exact weight_jit(knobs) per call, binned per observable.
    Covariances are block-diagonal across observables (independent measurements).  Returns a dict of fns +
    the stacked (data, Cinv) for external use.

    IMPORTANT (constant-folding NaN): the per-event weight MUST be evaluated as BR.weight_jit(JBs, knobs,
    grids) with JBs/grids as jit OPERANDS.  If JBs is instead baked into a jit as a closed-over constant,
    XLA constant-folds the 21M-row FSI gather and returns NaN (value-dependent).  So model_vec/chi2 below
    are deliberately NOT wrapped in jax.jit (that would re-capture JBs); the speed comes from weight_jit,
    and jax.grad/jax.hessian trace THROUGH weight_jit with JBs still an operand."""
    idxs = [s["idx"] for s in obs_specs]; nbs = [s["nb"] for s in obs_specs]
    bws = [s["bw"] for s in obs_specs]; convs = [s["conv"] for s in obs_specs]
    datas = [np.asarray(s["data"]) for s in obs_specs]
    cinvs, conds, nfl = [], [], []
    for s in obs_specs:
        ci, cond, nf = robust_cinv(s["cov"], rtol=rtol); cinvs.append(ci); conds.append(cond); nfl.append(nf)
    data_all = jnp.asarray(np.concatenate(datas))
    cinv_blk = np.zeros((len(data_all), len(data_all)))
    o = 0
    for ci in cinvs:
        n = ci.shape[0]; cinv_blk[o:o + n, o:o + n] = ci; o += n
    cinv_all = jnp.asarray(cinv_blk)

    def model_vec(knobs):
        w = BR.weight_jit(JBs, knobs, grids)                # JBs/grids are OPERANDS (see docstring) -> finite
        hs = []
        for idx, nb, bw, conv in zip(idxs, nbs, bws, convs):
            h = jax.ops.segment_sum(w, idx, num_segments=nb) / bw * conv
            hs.append(h)
        return jnp.concatenate(hs)

    def chi2_abs(knobs):
        r = model_vec(knobs) - data_all
        return r @ cinv_all @ r

    def chi2_shape(knobs):
        m = model_vec(knobs)
        A = (m @ cinv_all @ data_all) / (m @ cinv_all @ m)
        r = A * m - data_all
        return r @ cinv_all @ r

    def chi2_both(knobs):
        """One weight_jit pass -> (chi2_abs, chi2_shape, A_shape).  Halves grid cost vs calling both."""
        m = model_vec(knobs)
        ra = m - data_all
        A = (m @ cinv_all @ data_all) / (m @ cinv_all @ m)
        rs = A * m - data_all
        return ra @ cinv_all @ ra, rs @ cinv_all @ rs, A

    return dict(model_vec=model_vec, chi2_abs=chi2_abs, chi2_shape=chi2_shape, chi2_both=chi2_both,
                data_all=data_all, cinv_all=cinv_all, conds=conds, nfloored=nfl,
                ndf=len(data_all), obs_names=[s["obs"] for s in obs_specs],
                nbs=nbs, datas=datas, cinvs=cinvs, convs=convs, edges=[s["edges"] for s in obs_specs],
                derrs=[s["derr"] for s in obs_specs])


# ---- knob-vector <-> full dict for a chosen subset of SCALAR knobs ----------------------------- #
def assembler(names):
    """Return assemble(p) that overrides the named SCALAR knobs in nominal_knobs by p (a jnp vector)."""
    NOM = nominal_knobs()
    for n in names:
        assert not isinstance(NOM[n], tuple), f"{n} is a tuple knob; scalar pairs only here"

    def assemble(p):
        k = dict(NOM)
        for i, n in enumerate(names):
            k[n] = p[i]
        return k
    return assemble, NOM
