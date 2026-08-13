"""Information-content demonstrator: per-bin/per-dataset Fisher decomposition of a DIFFERENTIABLE generator.

Shows concretely that a differentiable model gives more than a scalar chi2 -- you can read off, per bin and
per dataset, which parameters carry the information.  The frozen 1M event bank + bank_reweight give an EXACT,
JAX-differentiable per-event weight w(theta), so d(dsigma/dx_bin)/dtheta_k is autodiff (NOT finite
difference).  We use that to

  (a) JOINTLY fit the selected params to TWO T2K STV channels at once with the full covariances, and
  (b) compute the exact per-bin Jacobian J = d(dsigma/dx_bin)/dtheta and the Fisher information J^T C^-1 J,
      decomposed by dataset and by bin.

Parameter sets (--set):
  axial4 (default) : M_A_qe, axial_strength, res_axial_strength, pion_pole  (the original 4-param demo;
                     M_A is now split per channel, so the cross-channel axial-mass knob is M_A_qe/M_A_res)
  full17           : M_A_qe, M_A_res + the 4 SF knobs (kF_sf, Eb_shift, sf_norm, src_tail) + the 11 FSI
                     knobs (sabs, s_piN_elastic, s_piN_cex, s_conv, s_NN_elastic[3], s_NN_inelastic[3],
                     f_NN_cex)

Five histograms across two channels, all with covariance, all sourced from the ONE bank (no regeneration):
  CC0pi-Np STV: dpt, dat            (signal_cc0pi, per-nucleon 1e-38 units; carbon only, no free-H)
  CC1pi+Np STV: pN, dpTT, daT       (signal_cc1pi_stv, nb-per-CH units; + FROZEN nominal free-H offset)

The CC1pi free-H piece (pi+ always survives, no nuclear FSI) is generated once at nominal and added as a
constant per bin -> its (small) knob dependence is neglected; the Jacobian for CC1pi is carbon-only.
This is a documented limitation of the demonstrator (logbook), not of the method.

    python -u analysis/t2k/differentiability/info_content.py [--set axial4|full17]
    python -u analysis/t2k/differentiability/info_content.py --plot-only /tmp/adonis_tune_runs/info_content_<set>.npz
"""
import os, sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from adonis.reweight import bank_plot as BP, bank_reweight as BR
from adonis.reweight.reweight_model import nominal_knobs, _EB_EPS

BANKDIR = os.environ.get("ADONIS_EVENT_BANK", "output/event_bank")

# ---- parameter specs: (knob_key, tuple_idx|None, tex label, clip_lo, clip_hi, closure theta*) ---------- #
PSETS = {
    "axial4": [
        ("M_A_qe",             None, r"$M_A^{\rm QE}$",          0.3, 3.0, 1.15),
        ("axial_strength",     None, r"$g_A^{\rm QE}$",          0.3, 3.0, 0.90),
        ("res_axial_strength", None, r"$g_A^{\rm RES}$",         0.3, 3.0, 1.20),
        ("pion_pole",          None, r"pion-pole",               0.3, 3.0, 0.80),
    ],
    "full17": [
        ("M_A_qe",             None, r"$M_A^{\rm QE}$",          0.3, 3.0, 1.15),
        ("M_A_res",            None, r"$M_A^{\rm RES}$",         0.3, 3.0, 0.85),
        ("kF_sf",              None, r"$k_F$",                   0.7, 1.4, 1.10),
        ("Eb_shift",           None, r"$E_b$ [MeV]",         _EB_EPS, 30.0, 5.00),
        ("sf_norm",            None, r"SF norm",                 0.3, 3.0, 0.95),
        ("src_tail",           None, r"SRC tail",                0.3, 3.0, 1.30),
        ("sabs",               None, r"$\sigma_{\pi\,\rm abs}$", 0.3, 3.0, 1.30),
        ("s_piN_elastic",      None, r"$\sigma_{\pi N\,\rm el}$",0.3, 3.0, 0.80),
        ("s_piN_cex",          None, r"$\sigma_{\pi N\,\rm cex}$",0.3, 3.0, 1.20),
        ("s_conv",             None, r"$\sigma_{\rm conv}$",     0.3, 3.0, 0.90),
        ("s_NN_elastic",       0,    r"$\sigma_{NN\,\rm el}$[pp]",0.3, 3.0, 1.20),
        ("s_NN_elastic",       1,    r"$\sigma_{NN\,\rm el}$[pn]",0.3, 3.0, 0.80),
        ("s_NN_elastic",       2,    r"$\sigma_{NN\,\rm el}$[nn]",0.3, 3.0, 1.10),
        ("s_NN_inelastic",     0,    r"$\sigma_{NN\,\rm inel}$[pp]",0.3, 3.0, 0.90),
        ("s_NN_inelastic",     1,    r"$\sigma_{NN\,\rm inel}$[pn]",0.3, 3.0, 1.15),
        ("s_NN_inelastic",     2,    r"$\sigma_{NN\,\rm inel}$[nn]",0.3, 3.0, 0.85),
        ("f_NN_cex",           None, r"$f_{NN\,\rm cex}$",       0.0, 1.0, 0.60),
    ],
}

PSET = "axial4"
for i, a in enumerate(sys.argv):
    if a == "--set":
        PSET = sys.argv[i + 1]
SPEC = PSETS[PSET]
PARAMS = [s[2] for s in SPEC]                 # tex labels (display)
PNAMES = [f"{s[0]}[{s[1]}]" if s[1] is not None else s[0] for s in SPEC]   # plain names (log/npz)
NPAR = len(SPEC)
_NOM = nominal_knobs()
def _nom_val(key, idx):
    v = getattr(_NOM, key)
    return float(v[idx]) if idx is not None else float(v)
THETA_NOM = jnp.asarray([_nom_val(s[0], s[1]) for s in SPEC])
THETA_STAR = np.array([s[5] for s in SPEC])
CLIP_LO = np.array([s[3] for s in SPEC]); CLIP_HI = np.array([s[4] for s in SPEC])
NPZ = f"/tmp/adonis_tune_runs/info_content_{PSET}.npz"
FIGTAG = "" if PSET == "axial4" else f"_{PSET}"    # axial4 keeps the original figure names

# CC1pi acceptance (thread the SINGLE source of truth: bank_plot's windows + forward cut) + CH normalization
_MU = (BP._MU_LO, BP._MU_HI); _PI = (BP._PI_LO, BP._PI_HI); _P = (BP._P_LO, BP._P_HI); _CTH = BP._CTH
NB_PER_CM2 = 1e33; A_CH = 13.0


# ------------------------------------------------------------------ data loaders ------------------------- #
def load_cc0pi(obs):
    """T2K CC0pi-Np STV data (edges, per-bin conv, data[1e-38], cov[1e-38^2]) -- as in tune.py."""
    r = uproot.open(f"../nuisance/data/T2K/CC0pi/STV/{'dpt' if obs == 'dpt' else 'dat'}Results.root")
    if obs == "dpt":
        edges = np.asarray(r["Result"].axis().edges()) * 1000.0             # MeV
        conv = 1e-33 / 12.0 * 1000.0 * 1e38
    else:
        edges = np.asarray(r["Result"].axis().edges())                     # rad
        conv = 1e-33 / 12.0 * 1e38
    data = np.asarray(r["Result"].values()) * 1e38
    cov = np.asarray(r["Covariance_Matrix"].values())
    return edges, conv, data, cov


def load_cc1pi(name):
    """T2K CC1pi+Np STV data (edges, data[nb/unit per CH], cov) -- as in make_plots.block_cc1pi_stv."""
    lines = open(f"../nuisance/data/T2K/CC1pipNp_STV/xsec_{name}.txt").read().splitlines()
    edges = np.array([float(x) for x in lines[0].split(":")[1].split()])
    vals = np.array([float(x) for x in lines[1].split(":")[1].split()]); nb = len(vals)
    cov = np.array([[float(x) for x in lines[3 + i].split()] for i in range(nb)])
    return edges, vals * NB_PER_CM2 * A_CH, cov * (NB_PER_CM2 * A_CH) ** 2


def freeH_offsets(edges_by_name):
    """Frozen nominal free-H (carbon+H -> CH) contribution per CC1pi observable, nb/unit.  pi+ always
    survives (no nuclear FSI); acceptance mirrors signal_cc1pi_stv (windows + cos70 forward cut); observables via the
    validated O.tki (NUISANCE hydrogen daT randomization included)."""
    from adonis.channels.free_proton import generate_H
    from adonis import kinematics as O
    from adonis.workflow.data_overlay import hydrogen_daT
    NH, NSEED = 50000, 4
    acc = {"pn": [], "dptt": [], "daT": []}; wl = []
    def _acc(p4, lo, hi):
        m = np.linalg.norm(p4[:, 1:], axis=1)
        return (m >= lo) & (m < hi) & (p4[:, 3] / np.clip(m, 1e-9, None) > _CTH)
    for s in range(NSEED):
        knu, kmu, pN, pPi, w = (np.asarray(x) for x in generate_H(NH, seed=s))
        m = (w > 0) & _acc(kmu, *_MU) & _acc(pPi, *_PI) & _acc(pN, *_P)
        dptt, pn, dat, _ = O.tki(kmu[m], pPi[m], pN[m])
        dat = hydrogen_daT(dptt, dat, np.ones(int(m.sum()), bool), s)
        acc["pn"].append(np.asarray(pn)); acc["dptt"].append(np.asarray(dptt)); acc["daT"].append(np.degrees(dat)); wl.append(w[m])
    W = np.concatenate(wl) / NSEED
    vals = {k: np.concatenate(v) for k, v in acc.items()}
    out = {}
    for name, edges in edges_by_name.items():
        h, _ = np.histogram(vals[name], bins=edges, weights=W)
        out[name] = h / np.diff(edges)                                     # nb/unit
    print(f"  free-H: {len(W)} accepted ev  sigma={W.sum():.4e} nb", flush=True)
    return out


# ------------------------------------------------------------------ dataset assembly --------------------- #
def _bin(sig_mask, values, edges):
    """(sel_idx, binidx, nbin): keep signal events with values in [edges[0], edges[-1]); assign bin."""
    nbin = len(edges) - 1
    idx = np.searchsorted(edges, values) - 1
    inb = sig_mask & (values >= edges[0]) & (values < edges[-1])
    sel = np.where(inb)[0]
    return sel.astype(np.int64), idx[sel].astype(np.int64), nbin


def build_datasets(B):
    """List of dataset dicts: name, channel, sel_idx, binidx, nbin, scale_bin, offset, data, Cinv, sigma,
    edges, xlabel, xscale (plot units)."""
    ds = []
    lead0, _ = BP.leading_proton(B); sig0 = BP.signal_cc0pi(B)[0]
    kmu = B["k_lep"].astype(np.float64)
    # ---- CC0pi dpt / dat (carbon, per-nucleon 1e-38) ----
    for obs, valfn, xlab, xsc in (
            ("dpt", lambda: np.asarray(BP.dpt(B, lead0)), r"$\delta p_T$ [GeV/c]", 1000.0),
            ("dat", lambda: np.asarray(BP.dat(B, lead0)), r"$\delta\alpha_T$ [rad]", 1.0)):
        edges, conv, data, cov = load_cc0pi(obs)
        sel, bidx, nb = _bin(sig0, valfn(), edges)
        Cinv = np.linalg.inv(cov + 1e-12 * np.eye(nb))
        ds.append(dict(name=f"CC0pi {obs}", channel="CC0pi", key=obs, sel_idx=sel, binidx=bidx, nbin=nb,
                       scale_bin=conv / np.diff(edges), offset=np.zeros(nb), data=data, Cinv=Cinv,
                       sigma=np.sqrt(np.diag(cov)), edges=edges, xlabel=xlab, xscale=xsc))
    # ---- CC1pi pN / dpTT / daT (carbon bank + frozen free-H, nb/unit per CH) ----
    mask1, lead1, pip1 = BP.signal_cc1pi_stv(B)
    cc1 = [("pN", "pn", lambda: np.asarray(BP.pN_1pi(kmu, lead1, pip1)), r"$p_N$ [MeV/c]", 1.0),
           ("dpTT", "dptt", lambda: np.asarray(BP.dptt_1pi(kmu, lead1, pip1)), r"$\delta p_{TT}$ [MeV/c]", 1.0),
           ("daT", "daT", lambda: np.degrees(np.asarray(BP.dat_1pi(kmu, lead1, pip1))), r"$\delta\alpha_T$ [deg]", 1.0)]
    edges_by = {}
    tmp = []
    for obs, dkey, valfn, xlab, xsc in cc1:
        edges, data, cov = load_cc1pi(obs)
        edges_by[dkey] = edges
        tmp.append((obs, dkey, valfn, xlab, xsc, edges, data, cov))
    fH = freeH_offsets(edges_by)
    for obs, dkey, valfn, xlab, xsc, edges, data, cov in tmp:
        sel, bidx, nb = _bin(mask1, valfn(), edges)
        Cinv = np.linalg.inv(cov + 1e-6 * np.max(np.diag(cov)) * np.eye(nb))
        ds.append(dict(name=f"CC1pi {obs}", channel="CC1pi", key=dkey, sel_idx=sel, binidx=bidx, nbin=nb,
                       scale_bin=1.0 / np.diff(edges), offset=fH[dkey], data=data, Cinv=Cinv,
                       sigma=np.sqrt(np.diag(cov)), edges=edges, xlabel=xlab, xscale=xsc))
    return ds


# ------------------------------------------------------------------ differentiable model ----------------- #
# JB (the ~2.7 GB bank pytree) is threaded as an explicit ARGUMENT so it is a function argument -- never a
# captured jit constant.  We differentiate only w.r.t. theta (argnums=0), exactly as production weight_jit
# keeps JB out of the compiled constant pool.  grids / nominal are small and closed over.

def knobs_of(theta, nominal):
    """Assemble a full PhysicsParams from the SPEC-selected theta vector (tuple components rebuilt)."""
    upd = {}
    tup = {}
    for i, (key, idx, *_rest) in enumerate(SPEC):
        if idx is None:
            upd[key] = theta[i]
        else:
            tup.setdefault(key, list(getattr(nominal, key)))[idx] = theta[i]
    for key, comps in tup.items():
        upd[key] = tuple(comps)
    return nominal._replace(**upd)


# ---------------------------------------------------------------------------------------------------
# ONE binning primitive for every sample type.
#
# Both sample families reduce per-event weights to per-bin observables with the SAME linear map,
#
#     m_b = scale_b * sum_{e : binidx_e = b} coef_e * w_[sel_e]   (+ offset_b)
#
# and used to implement it twice with np.bincount:
#     BankSample : scale_bin * bincount(binidx, w[sel_idx])          + free-H offset
#     BeamSample : (piR^2/n_tried) * bincount(idx, coef*w)           coef = 1 (reaction) or `second`
# The only differences are an event selection in one and a per-event coefficient in the other -- neither
# is exclusive, and neither is physics.  Duplicating it meant a device-resident/JAX binning could only
# ever cover part of the model, which in turn made the whole model NOT reverse-mode differentiable: a
# scalar gradient still cost one forward JVP per dial instead of a single VJP.
#
# `apply_dev` keeps the reduction on device, so a model evaluation returns ~nbin floats instead of
# pulling ~1.9M per-event weights back to the host, AND is differentiable end to end.  Events are sorted
# by bin ONCE at construction so segment_sum uses a segmented reduction rather than contended atomics
# (~6000 events/bin here).
# ---------------------------------------------------------------------------------------------------
class BinSpec:
    """The fixed sparse map w -> per-bin observable, shared by every sample type.

    Two index orderings are kept deliberately:
      HOST   events in BANK order.  np.bincount does not care about order, but the gather w[sel] does --
             bin-sorting it scatters the reads and measured 2.06x slower (~52 ms per model evaluation
             across the 19 datasets).  So the host path keeps the natural order.
      DEVICE events sorted by bin, so segment_sum uses a segmented reduction instead of contended
             atomics (~6000 events/bin here).  Built lazily; nothing pays for it unless S4_JAX_BIN=1.
    """

    __slots__ = ("sel", "coef", "binidx", "nbin", "scale", "offset", "_dev")

    def __init__(self, binidx, nbin, scale, sel=None, coef=None, offset=None):
        n = len(np.asarray(binidx))
        self.binidx = np.asarray(binidx, np.int64)
        self.sel = np.arange(n, dtype=np.int64) if sel is None else np.asarray(sel, np.int64)
        self.coef = None if coef is None else np.asarray(coef, float)
        self.nbin = int(nbin)
        self.scale = np.asarray(scale, float)
        self.offset = None if offset is None else np.asarray(offset, float)
        self._dev = None

    def apply(self, w):
        """Host (numpy) reference path -- bank-ordered gather."""
        v = np.asarray(w)[self.sel]
        if self.coef is not None:
            v = self.coef * v
        out = self.scale * np.bincount(self.binidx, weights=v, minlength=self.nbin)
        return out if self.offset is None else out + self.offset

    def _device(self):
        if self._dev is None:
            import jax.numpy as jnp
            o = np.argsort(self.binidx, kind="stable")      # bin-sorted ONLY for the device reduction
            self._dev = dict(binidx=jnp.asarray(self.binidx[o]), sel=jnp.asarray(self.sel[o]),
                             coef=None if self.coef is None else jnp.asarray(self.coef[o]),
                             scale=jnp.asarray(self.scale),
                             offset=None if self.offset is None else jnp.asarray(self.offset))
        return self._dev

    def apply_dev(self, w):
        """Device path: identical arithmetic, stays on the GPU, reverse-mode differentiable."""
        import jax
        d = self._device()
        v = w[d["sel"]]
        if d["coef"] is not None:
            v = d["coef"] * v
        out = d["scale"] * jax.ops.segment_sum(v, d["binidx"], num_segments=self.nbin,
                                               indices_are_sorted=True)
        return out if d["offset"] is None else out + d["offset"]


def spec_of(d):
    """BinSpec for a BankSample-style dataset dict (cached on the dict)."""
    sp = d.get("_spec")
    if sp is None:
        sp = BinSpec(d["binidx"], d["nbin"], d["scale_bin"], sel=d["sel_idx"])
        d["_spec"] = sp
    return sp


def bin_w0(d, w):
    """Per-bin sum of a per-event quantity w for dataset d, scaled -- NO free-H offset (for the
    Jacobian: the frozen free-H offset is theta-independent so its derivative is zero)."""
    return spec_of(d).apply(w)


def bin_w(d, w):
    """Per-bin dsigma/dx for dataset d from a full per-event weight vector w (+ frozen free-H)."""
    return bin_w0(d, w) + d["offset"]


def bin_w0_dev(d, w):
    return spec_of(d).apply_dev(w)


def bin_w_dev(d, w):
    import jax.numpy as jnp
    return bin_w0_dev(d, w) + jnp.asarray(d["offset"])


# ------------------------------------------------------------------ main --------------------------------- #
def main():
    t0 = time.time()
    import resource
    def log(m):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30   # GB (peak RSS, macOS: bytes)
        print(f"[{time.time()-t0:6.1f}s|{rss:4.1f}G] {m}", flush=True)
    os.makedirs("/tmp/adonis_tune_runs", exist_ok=True); os.makedirs("output/figures", exist_ok=True)

    B = BP.load_bank(BANKDIR); JB = BR.to_jax(B); grids = BR.default_grids()
    nominal = nominal_knobs()
    log(f"bank {len(B['w0'])} ev ({(B['channel']==0).sum()} QE + {(B['channel']==1).sum()} RES)")
    log(f"param set '{PSET}': {NPAR} params: {PNAMES}")

    ds = build_datasets(B)
    for d in ds:
        d["Cinv_j"] = jnp.asarray(d["Cinv"])
    log(f"{len(ds)} datasets built")

    # ---- ONE differentiable weight function; JB is an ARGUMENT (never a jit constant) ----------------- #
    def wf(theta, JB):
        return BR.bank_weight(JB, knobs_of(theta, nominal), grids)
    wf_jit = jax.jit(wf)
    # per-EVENT weight Jacobian dw_e/dtheta, ONE KNOB AT A TIME via jvp: a full jacfwd carries an NPAR-wide
    # tangent on every (N,) / (N,cap) intermediate of the FSI-record graph (~16 GB at NPAR=17) -> OOM.
    # jvp keeps peak memory at ~2x one forward pass; compute is the same 17 forward passes jacfwd would do.
    jvp_wf = jax.jit(lambda tang, JB: jax.jvp(lambda th: wf(th, JB), (THETA_NOM,), (tang,))[1])
    w_nom = np.asarray(wf_jit(THETA_NOM, JB))
    CKPATH = NPZ.replace(".npz", "_grad.npz")
    SKIP_GRAD = "--skip-grad" in sys.argv                     # fits-only restart: gradient stage from checkpoint
    if SKIP_GRAD:
        assert os.path.exists(CKPATH), f"--skip-grad needs the checkpoint {CKPATH}"
        CK = dict(np.load(CKPATH, allow_pickle=True))
        assert str(CK["pset"]) == PSET
        for i, d in enumerate(ds):
            d["J"] = CK[f"J_{i}"]; d["Jw2"] = CK[f"Jw2_{i}"]; d["nom"] = bin_w(d, w_nom)
        content = CK["content"]; F_tot = CK["F_tot"]; Vpar = CK["Vpar"]
        ew = CK["fisher_eigvals"]; ev = CK["fisher_eigvecs"]
        fd_knob = CK["fd_knob"]; fd_knob_x = CK["fd_knob_x"]
        fd_max = float(fd_knob.max()); fd_max_x = float(fd_knob_x.max()); id_err = float(CK["id_err"])
        log(f"--skip-grad: gradient/Fisher stage restored from {CKPATH}")
    else:
        Jw_ev = np.empty((len(w_nom), NPAR))
        for k in range(NPAR):
            Jw_ev[:, k] = np.asarray(jvp_wf(jnp.zeros(NPAR).at[k].set(1.0), JB))
        log(f"per-event weight + Jacobian evaluated (Jw_ev {Jw_ev.shape}, jvp per knob)")

    if not SKIP_GRAD:
        for d in ds:
            d["nom"] = bin_w(d, w_nom)
            d["J"] = np.stack([bin_w0(d, Jw_ev[:, k]) for k in range(NPAR)], axis=1)   # (nbin, NPAR), no offset
            r = d["nom"] - d["data"]; chi2 = float(r @ d["Cinv"] @ r)
            log(f"  {d['name']:12s} nbin={d['nbin']}  nsig={len(d['sel_idx']):6d}  "
                f"nominal chi2/ndf={chi2/d['nbin']:.2f}")

        # ---- validation ladder ------------------------------------------------------------------------ #
        log("=== VALIDATION ===")
        # (1) nominal identity: bank_weight(theta_nom) reproduces the production weight_jit(nominal)
        w_direct = np.asarray(BR.weight_jit(JB, nominal, grids))
        id_err = float(np.max(np.abs(w_nom - w_direct)))
        log(f"(1) nominal identity |w(theta_nom) - w_nominal|_max = {id_err:.2e}")

        # (2) per-bin Jacobian: autodiff vs central finite-difference, reported PER KNOB.  Two columns:
        #     all events, and excluding the near-singular SF-edge population (per-event |J| > 100 x its
        #     knob's p99.999 -- the S(p,E)~0 clamp events where FD itself is non-convergent; sf_knob_checks).
        eps = 1e-3
        thr = np.percentile(np.abs(Jw_ev), 99.999, axis=0)
        sing = (np.abs(Jw_ev) > np.maximum(100 * thr, 1e-12)[None, :]).any(axis=1)
        log(f"(2) near-singular SF-edge events flagged: {int(sing.sum())} of {len(sing)}")
        fd_knob = np.zeros(NPAR); fd_knob_x = np.zeros(NPAR)
        for k in range(NPAR):
            tp = np.array(THETA_NOM); tm = np.array(THETA_NOM); tp[k] += eps; tm[k] -= eps
            fd_e = (np.asarray(wf_jit(jnp.asarray(tp), JB)) - np.asarray(wf_jit(jnp.asarray(tm), JB))) / (2 * eps)
            for d in ds:
                Jb = bin_w0(d, Jw_ev[:, k]); Fb = bin_w0(d, fd_e)
                sc = max(np.max(np.abs(Jb)), 1e-300)
                fd_knob[k] = max(fd_knob[k], float(np.max(np.abs(Jb - Fb)) / sc))
                JbX = bin_w0(d, np.where(sing, 0.0, Jw_ev[:, k])); FbX = bin_w0(d, np.where(sing, 0.0, fd_e))
                scX = max(np.max(np.abs(JbX)), 1e-300)
                fd_knob_x[k] = max(fd_knob_x[k], float(np.max(np.abs(JbX - FbX)) / scX))
            log(f"(2) {PNAMES[k]:22s} AD-vs-FD max rel = {fd_knob[k]:.2e}   (excl SF-edge: {fd_knob_x[k]:.2e})")
        fd_max = float(fd_knob.max()); fd_max_x = float(fd_knob_x.max())
        log(f"(2) worst AD-vs-FD rel err = {fd_max:.2e} (all events)  {fd_max_x:.2e} (excl SF-edge)  "
            f"({'PASS' if fd_max_x < 1e-2 else 'FAIL'} on the excl-edge gate)")

        # ---- Fisher decomposition --------------------------------------------------------------------- #
        log("=== FISHER ===")
        Fd = {}                                   # per-dataset NPARxNPAR Fisher
        for d in ds:
            J = d["J"]; Cinv = d["Cinv"]
            Fd[d["name"]] = J.T @ Cinv @ J
            L = np.linalg.cholesky(Cinv + 1e-15 * np.eye(d["nbin"]))          # Cinv = L L^T
            Jw = L.T @ J                                                       # F = Jw^T Jw
            d["Jw2"] = Jw ** 2
        F_tot = sum(Fd.values())
        Vpar = np.linalg.pinv(F_tot)              # pseudo-inverse: cov on the MEASURABLE subspace only
        content = np.array([[Fd[d["name"]][k, k] for k in range(NPAR)] for d in ds])   # (nds, NPAR)
        log(f"total-Fisher param sigmas (Cramer-Rao, pinv) = {np.sqrt(np.diag(Vpar))}")
        log(f"Fisher condition number = {np.linalg.cond(F_tot):.2e}")
        # eigen-spectrum: name the flat/degenerate directions (eigenvalue in a normalized basis so knobs
        # with different natural scales are comparable: F~ = D^-1/2 F D^-1/2)
        dg = np.sqrt(np.maximum(np.diag(F_tot), 1e-300))
        Fn = F_tot / np.outer(dg, dg)
        ew, ev = np.linalg.eigh(Fn)
        log("Fisher eigen-spectrum (normalized; small = flat direction):")
        for j in range(NPAR):
            top = np.argsort(-np.abs(ev[:, j]))[:4]
            comp = " ".join(f"{ev[t, j]:+.2f}*{PNAMES[t]}" for t in top)
            log(f"    lam={ew[j]:.3e}   {comp}")

        # ---- CHECKPOINT: gradient/Fisher stage persisted BEFORE the fits ------------------------------ #
        ck = dict(pset=PSET, params=np.array(PNAMES), plabels=np.array(PARAMS), theta_nom=np.array(THETA_NOM),
                  content=content, F_tot=F_tot, Vpar=Vpar, fisher_eigvals=ew, fisher_eigvecs=ev,
                  fd_knob=fd_knob, fd_knob_x=fd_knob_x, id_err=id_err, n_sing=int(sing.sum()),
                  ds_names=np.array([d["name"] for d in ds]), ds_channels=np.array([d["channel"] for d in ds]))
        for i, d in enumerate(ds):
            ck[f"J_{i}"] = d["J"]; ck[f"Jw2_{i}"] = d["Jw2"]; ck[f"nom_{i}"] = d["nom"]
            ck[f"data_{i}"] = d["data"]; ck[f"sigma_{i}"] = d["sigma"]; ck[f"edges_{i}"] = d["edges"]
            ck[f"xlabel_{i}"] = d["xlabel"]; ck[f"xscale_{i}"] = d["xscale"]
        np.savez(CKPATH, **ck)
        log(f"gradient/Fisher checkpoint saved -> {CKPATH}")
        Jw_ev = None                                           # (N,NPAR) no longer needed by the fits -> free it

    # ---- differentiable joint chi2 (JB threaded as argument; one bank pass per eval) ------------------ #
    STR = [dict(sel=jnp.asarray(d["sel_idx"]), bidx=jnp.asarray(d["binidx"]), nb=d["nbin"],
                scale=jnp.asarray(d["scale_bin"]), off=jnp.asarray(d["offset"]), Ci=d["Cinv_j"]) for d in ds]

    def bin_jax(w, s):
        """Per-bin dsigma/dx from an already-computed full per-event weight vector (shared across datasets)."""
        return jax.ops.segment_sum(w[s["sel"]], s["bidx"], num_segments=s["nb"]) * s["scale"] + s["off"]

    # FIT ENGINE: Levenberg-Marquardt on the FORWARD-mode (jvp) per-bin Jacobian.  On this graph a forward
    # pass is ~0.3 s but a reverse (value_and_grad) pass is ~7 s (gather-heavy FSI/SF VJPs) -- Adam at 600
    # reverse iterations was ~1.5 h PER FIT.  LM does 17 jvps + a 17x17 solve per step and converges in
    # O(10) steps on this near-quadratic chi2; the Gauss-Newton Hessian at the BFP comes for free.
    def models_to(datas, profile):
        """(m_jit, jvp_m): jitted concatenated per-bin model M(theta) (nbins_tot,) and its directional
        derivative.  profile=True computes the per-dataset analytic norm A_d INSIDE -> jvp carries dA/dtheta."""
        dj = [jnp.asarray(x) for x in datas]
        def model(theta, JB):
            w = BR.bank_weight(JB, knobs_of(theta, nominal), grids)   # ONE bank pass, all 5 datasets
            outs = []
            for s, dv in zip(STR, dj):
                mv = bin_jax(w, s)
                if profile:
                    A = (mv @ s["Ci"] @ dv) / (mv @ s["Ci"] @ mv); mv = A * mv
                outs.append(mv)
            return jnp.concatenate(outs)
        m_jit = jax.jit(model)
        jvp_m = jax.jit(lambda th, tang, JB: jax.jvp(lambda t: model(t, JB), (th,), (tang,))[1])
        return m_jit, jvp_m

    import scipy.linalg as sla
    Ci_block = sla.block_diag(*[d["Cinv"] for d in ds])                # (nbins_tot, nbins_tot)

    def rail_mask(theta):
        return (np.abs(theta - CLIP_LO) < 1e-9) | (np.abs(theta - CLIP_HI) < 1e-9)

    def lm_fit(m_jit, jvp_m, data_c, theta0, maxit=40, tag=""):
        """Box-clipped LM.  Returns (theta, chi2, J_bfp).  Logs every step (cheap: O(10) steps)."""
        theta = np.array(theta0, float)
        r = np.asarray(m_jit(jnp.asarray(theta), JB)) - data_c
        c = float(r @ Ci_block @ r)
        lam = 1e-3; J = None
        for it in range(maxit):
            J = np.stack([np.asarray(jvp_m(jnp.asarray(theta), jnp.zeros(NPAR).at[k].set(1.0), JB))
                          for k in range(NPAR)], axis=1)               # (nbins_tot, NPAR)
            g = J.T @ Ci_block @ r; Hgn = J.T @ Ci_block @ J
            dH = np.diag(np.maximum(np.diag(Hgn), 1e-12 * np.max(np.diag(Hgn))))
            accepted = False
            for _ in range(12):
                try:
                    delta = -np.linalg.solve(Hgn + lam * dH, g)
                except np.linalg.LinAlgError:
                    delta = -np.linalg.pinv(Hgn + lam * dH) @ g
                th_new = np.clip(theta + delta, CLIP_LO, CLIP_HI)
                r_new = np.asarray(m_jit(jnp.asarray(th_new), JB)) - data_c
                c_new = float(r_new @ Ci_block @ r_new)
                if c_new < c:
                    accepted = True; break
                lam *= 10.0
            log(f"  [lm{tag}] it {it:2d}  chi2 {c:.4e} -> {c_new:.4e}  lam={lam:.1e}  "
                f"|step|={np.max(np.abs(th_new - theta)):.2e}  {'ACC' if accepted else 'REJ'}")
            if not accepted:
                break
            dc = c - c_new; theta, r, c = th_new, r_new, c_new
            lam = max(lam / 3.0, 1e-12)
            if dc < 1e-9 * max(c, 1.0) and it >= 3:
                break
        return theta, c, J

    # ---- pseudo-data closure (absolute): data = model(theta*) ----------------------------------------- #
    log("=== PSEUDO-DATA CLOSURE ===")
    w_star = np.asarray(wf_jit(jnp.asarray(THETA_STAR), JB))
    pdata_c = np.concatenate([bin_w(d, w_star) for d in ds])
    m_abs, jvp_abs = models_to([d["data"] for d in ds], profile=False)   # abs model reused for all abs fits
    bfp_ps, l_ps, _ = lm_fit(m_abs, jvp_abs, pdata_c, THETA_NOM, tag=":clo")
    log(f"injected theta*   = {THETA_STAR}")
    log(f"recovered theta   = {bfp_ps}   (final chi2={l_ps:.2e})")
    log(f"closure |bias|max  = {np.max(np.abs(bfp_ps - THETA_STAR)):.2e}  "
        f"(param: {PNAMES[int(np.argmax(np.abs(bfp_ps - THETA_STAR)))]})")

    # ---- real-data joint fit: absolute + profiled-norm ------------------------------------------------ #
    log("=== JOINT FIT (real T2K data) ===")
    data_c = np.concatenate([d["data"] for d in ds])
    nbins_tot = sum(d["nbin"] for d in ds)
    ndf = nbins_tot - NPAR
    r0 = np.asarray(m_abs(THETA_NOM, JB)) - data_c
    chi2_nom = float(r0 @ Ci_block @ r0)
    bfp_abs, l_abs, J_bfp = lm_fit(m_abs, jvp_abs, data_c, THETA_NOM, tag=":abs")
    # Gauss-Newton Hessian at the BFP: H = 2 J^T Cinv J -> V = 2 pinv(H) (tune.py convention; GN drops the
    # residual x curvature term -- standard at a converged fit, and exact for the linear-model part)
    H = 2.0 * (J_bfp.T @ Ci_block @ J_bfp)
    with np.errstate(invalid="ignore"):
        Vabs = 2.0 * np.linalg.pinv(H); sig_abs = np.sqrt(np.diag(Vabs))
        corr_abs = Vabs / np.outer(sig_abs, sig_abs)
    log(f"ABSOLUTE  nominal chi2/ndf={chi2_nom/ndf:.2f}  BFP chi2/ndf={l_abs/ndf:.2f}  (ndf={ndf})")
    rm = rail_mask(bfp_abs)
    for i, p in enumerate(PNAMES):
        tag = " RAIL" if rm[i] else ""
        log(f"    {p:22s} = {bfp_abs[i]:.4f} +/- {sig_abs[i]:.4f}{tag}")

    m_prof, jvp_prof = models_to([d["data"] for d in ds], profile=True)
    bfp_prof, l_prof, Jp_bfp = lm_fit(m_prof, jvp_prof, data_c, THETA_NOM, tag=":prof")
    Hp = 2.0 * (Jp_bfp.T @ Ci_block @ Jp_bfp)
    with np.errstate(invalid="ignore"):
        Vprof = 2.0 * np.linalg.pinv(Hp); sig_prof = np.sqrt(np.diag(Vprof))
    ndf_prof = nbins_tot - NPAR - len(ds)
    rp0 = np.asarray(m_prof(THETA_NOM, JB)) - data_c
    log(f"PROFILED  nominal chi2/ndf={float(rp0 @ Ci_block @ rp0)/ndf_prof:.2f}  "
        f"BFP chi2/ndf={l_prof/ndf_prof:.2f}  (ndf={ndf_prof})")
    rmp = rail_mask(bfp_prof)
    for i, p in enumerate(PNAMES):
        tag = " RAIL" if rmp[i] else ""
        log(f"    {p:22s} = {bfp_prof[i]:.4f} +/- {sig_prof[i]:.4f}{tag}")
    w_bf = np.asarray(wf_jit(jnp.asarray(bfp_abs), JB))       # absolute best-fit per-event weights (overlays)
    w_pf = np.asarray(wf_jit(jnp.asarray(bfp_prof), JB))      # profiled best-fit per-event weights (overlays)
    with np.errstate(invalid="ignore"):
        corr_fisher = Vpar / np.outer(np.sqrt(np.diag(Vpar)), np.sqrt(np.diag(Vpar)))

    # ---- persist -------------------------------------------------------------------------------------- #
    save = dict(pset=PSET, params=np.array(PNAMES), plabels=np.array(PARAMS),
                theta_nom=np.array(THETA_NOM), theta_star=THETA_STAR, bfp_ps=bfp_ps,
                clip_lo=CLIP_LO, clip_hi=CLIP_HI,
                content=content, F_tot=F_tot, Vpar=Vpar, corr_fisher=corr_fisher,
                cramer_rao_sig=np.sqrt(np.diag(Vpar)),
                bfp_abs=bfp_abs, Vabs=Vabs, corr_abs=corr_abs, sig_abs=sig_abs,
                chi2_nom=chi2_nom, chi2_abs=l_abs, ndf=ndf,
                bfp_prof=bfp_prof, Vprof=Vprof, sig_prof=sig_prof, chi2_prof=l_prof, ndf_prof=ndf_prof,
                ds_names=np.array([d["name"] for d in ds]),
                ds_channels=np.array([d["channel"] for d in ds]),
                fd_max=fd_max, fd_max_x=fd_max_x, fd_knob=fd_knob, fd_knob_x=fd_knob_x,
                fisher_eigvals=ew, fisher_eigvecs=ev, id_err=id_err)
    for i, d in enumerate(ds):
        save[f"J_{i}"] = d["J"]; save[f"Jw2_{i}"] = d["Jw2"]; save[f"nom_{i}"] = d["nom"]
        save[f"data_{i}"] = d["data"]; save[f"sigma_{i}"] = d["sigma"]; save[f"edges_{i}"] = d["edges"]
        save[f"xlabel_{i}"] = d["xlabel"]; save[f"xscale_{i}"] = d["xscale"]
        save[f"bf_abs_{i}"] = bin_w(d, w_bf)
        mvp = bin_w(d, w_pf); Ci = d["Cinv"]; Ad = float((mvp @ Ci @ d["data"]) / (mvp @ Ci @ mvp))
        save[f"bf_prof_{i}"] = Ad * mvp; save[f"A_prof_{i}"] = Ad
    np.savez(NPZ, **save)
    log(f"saved -> {NPZ}")

    make_figures(ds, save)
    log("figures written")
    print("\n=== SUMMARY ===")
    print(f"param set              : {PSET} ({NPAR} params)")
    print(f"nominal identity err   : {id_err:.2e}")
    print(f"worst AD-vs-FD rel err : {fd_max:.2e} (all)  {fd_max_x:.2e} (excl SF-edge)")
    print(f"pseudo-data |bias|max  : {np.max(np.abs(bfp_ps - THETA_STAR)):.2e}")
    print(f"absolute joint chi2/ndf: {chi2_nom/ndf:.2f} (nom) -> {l_abs/ndf:.2f} (BFP)")


# ------------------------------------------------------------------ figures ------------------------------ #
def make_figures(ds, S):
    names = [d["name"] for d in ds]
    plabels = [str(x) for x in S["plabels"]]
    npar = len(plabels)
    # ===== Fig 1: per-bin info-weighted gradient heatmaps (params x bins), one panel per observable ===== #
    ph = max(4.0, 0.55 * npar + 1.8)
    fig, axes = plt.subplots(2, 3, figsize=(16, 2 * ph)); axes = axes.ravel()
    for a, d in zip(axes, ds):
        J = d["J"]; G = J / d["sigma"][:, None]                            # info-weighted gradient (per bin)
        M = G.T                                                            # (param, bin)
        vmax = np.max(np.abs(M)) + 1e-300
        im = a.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_yticks(range(npar)); a.set_yticklabels(plabels, fontsize=7)
        a.set_xticks(range(d["nbin"])); a.set_xticklabels(range(1, d["nbin"] + 1), fontsize=7)
        a.set_xlabel(f"{d['xlabel']} bin"); a.set_title(f"{d['name']}   (info-wt. grad $J_{{ik}}/\\sigma_i$)", fontsize=9)
        for k in range(npar):                                             # mark each param's peak bin
            pk = int(np.argmax(np.abs(M[k]))); a.plot(pk, k, "k*", ms=7)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    for a in axes[len(ds):]:
        a.axis("off")
    fig.suptitle(r"Per-bin sensitivity $\partial(\mathrm{d}\sigma/\mathrm{d}x)_i/\partial\theta_k$ / $\sigma_i$ "
                 r"— every parameter touches ALL bins, but $\star$ marks where each pulls hardest", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(f"output/figures/info_content_gradient_heatmaps{FIGTAG}.png", dpi=130)
    plt.close(fig); print(f"wrote output/figures/info_content_gradient_heatmaps{FIGTAG}.png", flush=True)

    # ===== Fig 2: dataset x param Fisher content + stacked total info =================================== #
    content = S["content"]                                                 # (nds, npar)
    col_norm = content / (content.sum(axis=0, keepdims=True) + 1e-300)      # each param's info split by dataset
    fig, ax = plt.subplots(1, 2, figsize=(max(15, 1.0 * npar + 8), 6.5))
    im = ax[0].imshow(col_norm.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)   # params on Y for many params
    ax[0].set_yticks(range(npar)); ax[0].set_yticklabels(plabels, fontsize=8)
    ax[0].set_xticks(range(len(ds))); ax[0].set_xticklabels(names, fontsize=8, rotation=20)
    for i in range(len(ds)):
        for k in range(npar):
            ax[0].text(i, k, f"{col_norm[i, k]*100:.0f}", ha="center", va="center",
                       color="w" if col_norm[i, k] < 0.6 else "k", fontsize=7)
    ax[0].set_title("Fisher content share [%] $F^{(d)}_{kk}/\\sum_d F^{(d)}_{kk}$\n(row = one param's info split across datasets)", fontsize=10)
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    cols = plt.cm.tab10(np.linspace(0, 1, len(ds)))
    bottom = np.zeros(npar)
    for i, d in enumerate(ds):
        ax[1].bar(range(npar), content[i], bottom=bottom, color=cols[i], label=names[i])
        bottom += content[i]
    ax[1].set_xticks(range(npar)); ax[1].set_xticklabels(plabels, fontsize=8, rotation=60, ha="right")
    ax[1].set_yscale("log"); ax[1].set_ylabel("Fisher information $F_{kk}$ (diag)")
    ax[1].set_title("Total Fisher information per parameter (stacked by dataset)", fontsize=10)
    ax[1].legend(fontsize=8)
    fig.suptitle("Which dataset carries which parameter's information", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"output/figures/info_content_fisher_breakdown{FIGTAG}.png", dpi=130)
    plt.close(fig); print(f"wrote output/figures/info_content_fisher_breakdown{FIGTAG}.png", flush=True)

    # ===== Fig 3: joint-fit overlays (5 obs, absolute + profiled) + Fisher param correlation ============ #
    fig, axes = plt.subplots(2, 3, figsize=(16, 9)); axes = axes.ravel()
    for a, i in zip(axes, range(len(ds))):
        d = ds[i]; edges = S[f"edges_{i}"] / S[f"xscale_{i}"]; ctr = 0.5 * (edges[1:] + edges[:-1])
        a.errorbar(ctr, S[f"data_{i}"], yerr=S[f"sigma_{i}"], fmt="o", color="k", ms=4, capsize=2, label="T2K data")
        a.step(edges, np.append(S[f"nom_{i}"], S[f"nom_{i}"][-1]), where="post", color="0.5", ls=":", lw=1.5,
               label="ADoNIS nominal")
        a.step(edges, np.append(S[f"bf_abs_{i}"], S[f"bf_abs_{i}"][-1]), where="post", color="C0", lw=1.8,
               label="joint best fit (absolute)")
        if f"bf_prof_{i}" in S:
            a.step(edges, np.append(S[f"bf_prof_{i}"], S[f"bf_prof_{i}"][-1]), where="post", color="C3",
                   lw=1.4, ls="--", label=f"joint best fit (prof. norm, $A_d$={float(S[f'A_prof_{i}']):.2f})")
        a.set_xlabel(str(S[f"xlabel_{i}"])); a.set_title(d["name"], fontsize=10); a.set_ylim(bottom=0)
        a.legend(fontsize=7)
    # parameter correlation: Fisher (Cramer-Rao) at nominal -- always well-defined, unlike the Hessian at a rail
    a = axes[5]; corr = S["corr_fisher"]
    im = a.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    a.set_xticks(range(npar)); a.set_xticklabels(plabels, fontsize=6, rotation=90)
    a.set_yticks(range(npar)); a.set_yticklabels(plabels, fontsize=6)
    if npar <= 8:
        for i in range(npar):
            for k in range(npar):
                a.text(k, i, f"{corr[i, k]:+.2f}", ha="center", va="center",
                       color="w" if abs(corr[i, k]) > 0.5 else "k", fontsize=7)
    a.set_title("param correlation (Fisher $F^{-1}$ at nominal)", fontsize=10)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    bfp = S["bfp_abs"]; sig = S["sig_abs"]
    lo = S["clip_lo"]; hi = S["clip_hi"]
    rail = (np.abs(bfp - lo) < 1e-9) | (np.abs(bfp - hi) < 1e-9)
    if npar <= 6:
        err = [f"±{sig[i]:.2f}" if np.isfinite(sig[i]) and not rail[i] else " (rail)" for i in range(npar)]
        txt = "   [" + "  ".join(f"{str(S['params'][i])}={bfp[i]:.2f}{err[i]}" for i in range(npar)) + "]"
    else:
        txt = f"   ({int(rail.sum())} of {npar} params at a clip rail)"
    fig.suptitle(f"Joint {npar}-param fit to T2K CC0$\\pi$ + CC1$\\pi^+$ STV (absolute)\n"
                 f"$\\chi^2$/ndf {S['chi2_nom']/S['ndf']:.2f}$\\to${S['chi2_abs']/S['ndf']:.2f}{txt}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"output/figures/info_content_joint_fit{FIGTAG}.png", dpi=130)
    plt.close(fig); print(f"wrote output/figures/info_content_joint_fit{FIGTAG}.png", flush=True)


def plot_only(npz):
    global FIGTAG
    S = dict(np.load(npz, allow_pickle=True))
    ps = str(S.get("pset", "axial4"))
    FIGTAG = "" if ps == "axial4" else f"_{ps}"               # figure names follow the npz's param set
    ds = []
    n = len([k for k in S if k.startswith("J_")])
    for i in range(n):
        ds.append(dict(name=str(S["ds_names"][i]), channel=str(S["ds_channels"][i]), J=S[f"J_{i}"],
                       Jw2=S[f"Jw2_{i}"], sigma=S[f"sigma_{i}"], nbin=S[f"J_{i}"].shape[0],
                       edges=S[f"edges_{i}"], xlabel=str(S[f"xlabel_{i}"]), xscale=float(S[f"xscale_{i}"]),
                       nom=S[f"nom_{i}"], data=S[f"data_{i}"]))
    make_figures(ds, S)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot_only(sys.argv[sys.argv.index("--plot-only") + 1])
    else:
        main()
