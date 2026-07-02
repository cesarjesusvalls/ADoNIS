"""Information-content demonstrator: per-bin/per-dataset Fisher decomposition of a DIFFERENTIABLE generator.

Shows concretely that a differentiable model gives more than a scalar chi2 -- you can read off, per bin and
per dataset, which parameters carry the information.  The frozen 1M event bank + bank_reweight.weight_jit give
an EXACT, JAX-differentiable per-event weight w(theta) for all 26 knobs, so d(dsigma/dx_bin)/dtheta_k is
autodiff (NOT finite difference).  We use that to

  (a) JOINTLY fit 4 physics params to TWO T2K STV channels at once with the full covariances, and
  (b) compute the exact per-bin Jacobian J = d(dsigma/dx_bin)/dtheta and the Fisher information J^T C^-1 J,
      decomposed by dataset and by bin.

Four params, all nominal = 1.0 (multiplicative knobs already wired into bank_reweight.bank_weight):
  M_A                 axial mass          -- CROSS-CHANNEL (ma_reweight on BOTH qe_ma and res_ma)
  axial_strength      QE axial scale      -- QE-only
  res_axial_strength  RES axial scale     -- RES-only
  pion_pole           RES pion-pole term  -- RES-only

Five histograms across two channels, all with covariance, all sourced from the ONE bank (no regeneration):
  CC0pi-Np STV: dpt, dat            (signal_cc0pi, per-nucleon 1e-38 units; carbon only, no free-H)
  CC1pi+Np STV: pN, dpTT, daT       (signal_cc1pi_stv, nb-per-CH units; + FROZEN nominal free-H offset)

The CC1pi free-H piece (pi+ always survives, no nuclear FSI) is generated once at nominal and added as a
constant per bin -> its (small) RES-knob dependence is neglected; the Jacobian for CC1pi is carbon-only.
This is a documented limitation of the demonstrator (logbook), not of the method.

    python -u scripts/info_content.py            # full run
    python -u scripts/info_content.py --plot-only /tmp/adonis_tune_runs/info_content.npz
"""
import os, sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from analysis.t2k.differentiability import bank_plot as BP, bank_reweight as BR
from analysis.t2k.differentiability.full_knobs import nominal_knobs

BANKDIR = os.environ.get("ADONIS_EVENT_BANK", "output/event_bank")
NPZ = "/tmp/adonis_tune_runs/info_content.npz"

PARAMS = ["M_A", "axial_strength", "res_axial_strength", "pion_pole"]
PLABEL = [r"$M_A$", r"$g_A^{\rm QE}$", r"$g_A^{\rm RES}$", "pion-pole"]
NPAR = len(PARAMS)
THETA_NOM = jnp.array([1.0, 1.0, 1.0, 1.0])
THETA_STAR = np.array([1.15, 0.90, 1.20, 0.80])       # pseudo-data injection point (closure)

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
    from adonis.workflow.free_proton import generate_H
    import adonis.workflow.observables as O
    NH, NSEED = 50000, 4
    acc = {"pn": [], "dptt": [], "daT": []}; wl = []
    def _acc(p4, lo, hi):
        m = np.linalg.norm(p4[:, 1:], axis=1)
        return (m >= lo) & (m < hi) & (p4[:, 3] / np.clip(m, 1e-9, None) > _CTH)
    for s in range(NSEED):
        knu, kmu, pN, pPi, w = (np.asarray(x) for x in generate_H(NH, seed=s))
        m = (w > 0) & _acc(kmu, *_MU) & _acc(pPi, *_PI) & _acc(pN, *_P)
        dptt, pn, dat, _ = O.tki(kmu[m], pPi[m], pN[m], np.ones(int(m.sum()), bool), s)
        acc["pn"].append(pn); acc["dptt"].append(dptt); acc["daT"].append(np.degrees(dat)); wl.append(w[m])
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
    kmu = B["k_mu"].astype(np.float64)
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
    k = dict(nominal)
    k["M_A"] = theta[0]; k["axial_strength"] = theta[1]
    k["res_axial_strength"] = theta[2]; k["pion_pole"] = theta[3]
    return k


def bin_w0(d, w):
    """Per-bin sum of a per-event quantity w (numpy) for dataset d, scaled -- NO free-H offset (for the
    Jacobian: the frozen free-H offset is theta-independent so its derivative is zero)."""
    return d["scale_bin"] * np.bincount(d["binidx"], weights=np.asarray(w)[d["sel_idx"]], minlength=d["nbin"])


def bin_w(d, w):
    """Per-bin dsigma/dx for dataset d from a full per-event weight vector w (numpy) (+ frozen free-H)."""
    return bin_w0(d, w) + d["offset"]


# ------------------------------------------------------------------ main --------------------------------- #
def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    os.makedirs("/tmp/adonis_tune_runs", exist_ok=True); os.makedirs("output/figures", exist_ok=True)

    B = BP.load_bank(BANKDIR); JB = BR.to_jax(B); grids = BR.default_grids()
    nominal = nominal_knobs()
    log(f"bank {len(B['w0'])} ev ({(B['channel']==0).sum()} QE + {(B['channel']==1).sum()} RES)")

    ds = build_datasets(B)
    for d in ds:
        d["Cinv_j"] = jnp.asarray(d["Cinv"])
    log(f"{len(ds)} datasets built")

    # ---- ONE differentiable weight function; JB is an ARGUMENT (never a jit constant) ----------------- #
    def wf(theta, JB):
        return BR.bank_weight(JB, knobs_of(theta, nominal), grids)
    wf_jit = jax.jit(wf)
    jac_wf = jax.jit(jax.jacfwd(wf, argnums=0))               # per-EVENT weight Jacobian dw_e/dtheta

    w_nom = np.asarray(wf_jit(THETA_NOM, JB))
    Jw_ev = np.asarray(jac_wf(THETA_NOM, JB))                 # (N, 4) -- computed once, reused for all datasets
    log(f"per-event weight + Jacobian evaluated (Jw_ev {Jw_ev.shape})")

    for d in ds:
        d["nom"] = bin_w(d, w_nom)
        d["J"] = np.stack([bin_w0(d, Jw_ev[:, k]) for k in range(NPAR)], axis=1)   # (nbin, 4), no offset
        r = d["nom"] - d["data"]; chi2 = float(r @ d["Cinv"] @ r)
        log(f"  {d['name']:12s} nbin={d['nbin']}  nsig={len(d['sel_idx']):6d}  "
            f"nominal chi2/ndf={chi2/d['nbin']:.2f}")

    # ---- validation ladder ---------------------------------------------------------------------------- #
    log("=== VALIDATION ===")
    # (1) nominal identity: bank_weight(theta_nom) reproduces the production weight_jit(nominal)
    w_direct = np.asarray(BR.weight_jit(JB, nominal, grids))
    id_err = float(np.max(np.abs(w_nom - w_direct)))
    log(f"(1) nominal identity |w(theta_nom) - w_nominal|_max = {id_err:.2e}")

    # (2) per-bin Jacobian: autodiff vs central finite-difference (recompute weights at theta +/- eps)
    fd_report = {}
    eps = 1e-3
    wpm = {}
    for k in range(NPAR):
        tp = np.array(THETA_NOM); tm = np.array(THETA_NOM); tp[k] += eps; tm[k] -= eps
        wpm[k] = (np.asarray(wf_jit(jnp.asarray(tp), JB)), np.asarray(wf_jit(jnp.asarray(tm), JB)))
    for d in ds:
        J = d["J"]; Jfd = np.zeros_like(J)
        for k in range(NPAR):
            Jfd[:, k] = (bin_w0(d, wpm[k][0]) - bin_w0(d, wpm[k][1])) / (2 * eps)
        sc = np.maximum(np.max(np.abs(J), axis=0), 1e-300)
        rel = np.max(np.abs(J - Jfd) / sc[None, :])
        fd_report[d["name"]] = rel
        log(f"(2) {d['name']:12s} AD-vs-FD max rel err = {rel:.2e}")
    fd_max = max(fd_report.values())
    log(f"(2) worst AD-vs-FD rel err across datasets = {fd_max:.2e}  ({'PASS' if fd_max < 1e-2 else 'FAIL'})")

    # ---- Fisher decomposition ------------------------------------------------------------------------- #
    log("=== FISHER ===")
    Fd = {}                                   # per-dataset 4x4 Fisher
    perbin = {}                               # per-dataset whitened per-bin content Jw^2 (nbin,4)
    for d in ds:
        J = d["J"]; Cinv = d["Cinv"]
        Fd[d["name"]] = J.T @ Cinv @ J
        L = np.linalg.cholesky(Cinv + 1e-15 * np.eye(d["nbin"]))          # Cinv = L L^T
        Jw = L.T @ J                                                       # F = Jw^T Jw
        perbin[d["name"]] = Jw ** 2
        d["Jw2"] = Jw ** 2
    F_tot = sum(Fd.values())
    Vpar = np.linalg.inv(F_tot)               # Fisher-predicted param covariance (Cramer-Rao)
    # dataset x param content matrix (diagonal Fisher info per dataset)
    content = np.array([[Fd[d["name"]][k, k] for k in range(NPAR)] for d in ds])   # (nds, 4)
    log(f"total-Fisher param sigmas (Cramer-Rao) = {np.sqrt(np.diag(Vpar))}")

    # ---- differentiable joint chi2 (JB threaded as argument; one bank pass per eval) ------------------ #
    # precompute per-dataset static index/scale/offset/data/Cinv as device arrays (small; safe to close over)
    STR = [dict(sel=jnp.asarray(d["sel_idx"]), bidx=jnp.asarray(d["binidx"]), nb=d["nbin"],
                scale=jnp.asarray(d["scale_bin"]), off=jnp.asarray(d["offset"]), Ci=d["Cinv_j"]) for d in ds]

    def model_jax(theta, JB, s):
        w = BR.bank_weight(JB, knobs_of(theta, nominal), grids)
        return jax.ops.segment_sum(w[s["sel"]], s["bidx"], num_segments=s["nb"]) * s["scale"] + s["off"]

    def chi2_to(datas, profile):
        dj = [jnp.asarray(x) for x in datas]
        def chi2(theta, JB):
            c = 0.0
            for s, dv in zip(STR, dj):
                mv = model_jax(theta, JB, s)
                if profile:
                    A = (mv @ s["Ci"] @ dv) / (mv @ s["Ci"] @ mv); mv = A * mv
                r = mv - dv; c = c + r @ s["Ci"] @ r
            return c
        return jax.jit(chi2)

    def adam_fit(chi2, datas_ignored, theta0, niters=400, lr=0.02, lo=0.3, hi=3.0):
        vg = jax.jit(jax.value_and_grad(chi2, argnums=0))
        theta = jnp.asarray(theta0); mm = jnp.zeros(NPAR); vv = jnp.zeros(NPAR); traj = [np.array(theta)]
        l = 0.0
        for it in range(niters):
            l, g = vg(theta, JB)
            mm = 0.9 * mm + 0.1 * g; vv = 0.999 * vv + 0.001 * g ** 2
            mh = mm / (1 - 0.9 ** (it + 1)); vh = vv / (1 - 0.999 ** (it + 1))
            theta = jnp.clip(theta - lr * mh / (jnp.sqrt(vh) + 1e-12), lo, hi); traj.append(np.array(theta))
        return np.array(theta), np.array(traj), float(l)

    # ---- pseudo-data closure (absolute): data = model(theta*) ----------------------------------------- #
    log("=== PSEUDO-DATA CLOSURE ===")
    w_star = np.asarray(wf_jit(jnp.asarray(THETA_STAR), JB))
    pdata = [bin_w(d, w_star) for d in ds]
    chi2_ps = chi2_to(pdata, profile=False)
    bfp_ps, _, l_ps = adam_fit(chi2_ps, None, THETA_NOM)
    log(f"injected theta*   = {THETA_STAR}")
    log(f"recovered theta   = {bfp_ps}   (final chi2={l_ps:.2e})")
    log(f"closure |bias|max  = {np.max(np.abs(bfp_ps - THETA_STAR)):.2e}")

    # ---- real-data joint fit: absolute + profiled-norm ------------------------------------------------ #
    log("=== JOINT FIT (real T2K data) ===")
    chi2_abs = chi2_to([d["data"] for d in ds], profile=False)
    ndf = sum(d["nbin"] for d in ds) - NPAR
    chi2_nom = float(chi2_abs(THETA_NOM, JB))
    bfp_abs, traj_abs, l_abs = adam_fit(chi2_abs, None, THETA_NOM)
    H = np.asarray(jax.hessian(chi2_abs, argnums=0)(jnp.asarray(bfp_abs), JB))
    Vabs = 2.0 * np.linalg.inv(H); sig_abs = np.sqrt(np.diag(Vabs))
    corr_abs = Vabs / np.outer(sig_abs, sig_abs)
    log(f"ABSOLUTE  nominal chi2/ndf={chi2_nom/ndf:.2f}  BFP chi2/ndf={l_abs/ndf:.2f}  (ndf={ndf})")
    for i, p in enumerate(PARAMS):
        log(f"    {p:20s} = {bfp_abs[i]:.4f} +/- {sig_abs[i]:.4f}")

    chi2_prof = chi2_to([d["data"] for d in ds], profile=True)
    bfp_prof, _, l_prof = adam_fit(chi2_prof, None, THETA_NOM)
    Hp = np.asarray(jax.hessian(chi2_prof, argnums=0)(jnp.asarray(bfp_prof), JB))
    Vprof = 2.0 * np.linalg.inv(Hp); sig_prof = np.sqrt(np.diag(Vprof))
    ndf_prof = sum(d["nbin"] for d in ds) - NPAR - len(ds)
    log(f"PROFILED  nominal chi2/ndf={float(chi2_prof(THETA_NOM, JB))/ndf_prof:.2f}  "
        f"BFP chi2/ndf={l_prof/ndf_prof:.2f}  (ndf={ndf_prof})")
    for i, p in enumerate(PARAMS):
        log(f"    {p:20s} = {bfp_prof[i]:.4f} +/- {sig_prof[i]:.4f}")
    w_bf = np.asarray(wf_jit(jnp.asarray(bfp_abs), JB))       # absolute best-fit per-event weights (overlays)
    w_pf = np.asarray(wf_jit(jnp.asarray(bfp_prof), JB))      # profiled best-fit per-event weights (overlays)
    corr_fisher = Vpar / np.outer(np.sqrt(np.diag(Vpar)), np.sqrt(np.diag(Vpar)))   # Cramer-Rao correlation

    # ---- persist -------------------------------------------------------------------------------------- #
    save = dict(params=np.array(PARAMS), theta_star=THETA_STAR, bfp_ps=bfp_ps,
                content=content, F_tot=F_tot, Vpar=Vpar, corr_fisher=corr_fisher,
                cramer_rao_sig=np.sqrt(np.diag(Vpar)),
                bfp_abs=bfp_abs, Vabs=Vabs, corr_abs=corr_abs, sig_abs=sig_abs,
                chi2_nom=chi2_nom, chi2_abs=l_abs, ndf=ndf,
                bfp_prof=bfp_prof, Vprof=Vprof, sig_prof=sig_prof, chi2_prof=l_prof, ndf_prof=ndf_prof,
                ds_names=np.array([d["name"] for d in ds]),
                ds_channels=np.array([d["channel"] for d in ds]),
                fd_max=fd_max, id_err=id_err)
    for i, d in enumerate(ds):
        save[f"J_{i}"] = d["J"]; save[f"Jw2_{i}"] = d["Jw2"]; save[f"nom_{i}"] = d["nom"]
        save[f"data_{i}"] = d["data"]; save[f"sigma_{i}"] = d["sigma"]; save[f"edges_{i}"] = d["edges"]
        save[f"xlabel_{i}"] = d["xlabel"]; save[f"xscale_{i}"] = d["xscale"]
        save[f"bf_abs_{i}"] = bin_w(d, w_bf)
        mvp = bin_w(d, w_pf); Ci = d["Cinv"]; Ad = float((mvp @ Ci @ d["data"]) / (mvp @ Ci @ mvp))
        save[f"bf_prof_{i}"] = Ad * mvp; save[f"A_prof_{i}"] = Ad
    np.savez(NPZ, **save)
    log(f"saved -> {NPZ}")

    make_figures(ds, save, fd_report)
    log("figures written")
    print("\n=== SUMMARY ===")
    print(f"nominal identity err   : {id_err:.2e}")
    print(f"worst AD-vs-FD rel err : {fd_max:.2e}")
    print(f"pseudo-data |bias|max  : {np.max(np.abs(bfp_ps - THETA_STAR)):.2e}")
    print(f"absolute joint chi2/ndf: {chi2_nom/ndf:.2f} (nom) -> {l_abs/ndf:.2f} (BFP)")


# ------------------------------------------------------------------ figures ------------------------------ #
def make_figures(ds, S, fd_report):
    names = [d["name"] for d in ds]
    # ===== Fig 1: per-bin info-weighted gradient heatmaps (params x bins), one panel per observable ===== #
    fig, axes = plt.subplots(2, 3, figsize=(16, 8)); axes = axes.ravel()
    for a, d in zip(axes, ds):
        J = d["J"]; G = J / d["sigma"][:, None]                            # info-weighted gradient (per bin)
        M = G.T                                                            # (param, bin)
        vmax = np.max(np.abs(M)) + 1e-300
        im = a.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_yticks(range(NPAR)); a.set_yticklabels(PLABEL, fontsize=9)
        a.set_xticks(range(d["nbin"])); a.set_xticklabels(range(1, d["nbin"] + 1), fontsize=7)
        a.set_xlabel(f"{d['xlabel']} bin"); a.set_title(f"{d['name']}   (info-wt. grad $J_{{ik}}/\\sigma_i$)", fontsize=9)
        for k in range(NPAR):                                             # mark each param's peak bin
            pk = int(np.argmax(np.abs(M[k]))); a.plot(pk, k, "k*", ms=8)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    for a in axes[len(ds):]:
        a.axis("off")
    fig.suptitle(r"Per-bin sensitivity $\partial(\mathrm{d}\sigma/\mathrm{d}x)_i/\partial\theta_k$ / $\sigma_i$ "
                 r"— every parameter touches ALL bins, but $\star$ marks where each pulls hardest", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig("output/figures/info_content_gradient_heatmaps.png", dpi=130)
    plt.close(fig); print("wrote output/figures/info_content_gradient_heatmaps.png", flush=True)

    # ===== Fig 2: dataset x param Fisher content + whitened per-bin content ============================= #
    content = S["content"]                                                 # (nds, 4)
    col_norm = content / (content.sum(axis=0, keepdims=True) + 1e-300)      # each param's info split by dataset
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    im = ax[0].imshow(col_norm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax[0].set_xticks(range(NPAR)); ax[0].set_xticklabels(PLABEL, fontsize=10)
    ax[0].set_yticks(range(len(ds))); ax[0].set_yticklabels(names, fontsize=9)
    for i in range(len(ds)):
        for k in range(NPAR):
            ax[0].text(k, i, f"{col_norm[i, k]*100:.0f}%", ha="center", va="center",
                       color="w" if col_norm[i, k] < 0.6 else "k", fontsize=9)
    ax[0].set_title("Fisher content share $F^{(d)}_{kk}/\\sum_d F^{(d)}_{kk}$\n(column = one param's info split across datasets)", fontsize=10)
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    # stacked bars: total Fisher info per param, colored by dataset
    cols = plt.cm.tab10(np.linspace(0, 1, len(ds)))
    bottom = np.zeros(NPAR)
    for i, d in enumerate(ds):
        ax[1].bar(range(NPAR), content[i], bottom=bottom, color=cols[i], label=names[i])
        bottom += content[i]
    ax[1].set_xticks(range(NPAR)); ax[1].set_xticklabels(PLABEL, fontsize=10)
    ax[1].set_yscale("log"); ax[1].set_ylabel("Fisher information $F_{kk}$ (diag)")
    ax[1].set_title("Total Fisher information per parameter (stacked by dataset)", fontsize=10)
    ax[1].legend(fontsize=8)
    fig.suptitle("Which dataset carries which parameter's information", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig("output/figures/info_content_fisher_breakdown.png", dpi=130)
    plt.close(fig); print("wrote output/figures/info_content_fisher_breakdown.png", flush=True)

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
    a.set_xticks(range(NPAR)); a.set_xticklabels(PLABEL, fontsize=9, rotation=30)
    a.set_yticks(range(NPAR)); a.set_yticklabels(PLABEL, fontsize=9)
    for i in range(NPAR):
        for k in range(NPAR):
            a.text(k, i, f"{corr[i, k]:+.2f}", ha="center", va="center",
                   color="w" if abs(corr[i, k]) > 0.5 else "k", fontsize=8)
    a.set_title("param correlation (Fisher $F^{-1}$ at nominal)", fontsize=10)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    bfp = S["bfp_abs"]; sig = S["sig_abs"]
    # a param at the [0.3, 3] clip rail has an invalid Hessian error (blueprint caveat) -> label it 'rail'
    err = [f"±{sig[i]:.2f}" if np.isfinite(sig[i]) else " (rail)" for i in range(NPAR)]
    txt = "  ".join(f"{p}={bfp[i]:.2f}{err[i]}" for i, p in enumerate(PARAMS))
    fig.suptitle(f"Joint 4-param fit to T2K CC0$\\pi$ + CC1$\\pi^+$ STV (absolute)\n"
                 f"$\\chi^2$/ndf {S['chi2_nom']/S['ndf']:.1f}$\\to${S['chi2_abs']/S['ndf']:.1f}   [{txt}]", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig("output/figures/info_content_joint_fit.png", dpi=130)
    plt.close(fig); print("wrote output/figures/info_content_joint_fit.png", flush=True)


def plot_only(npz):
    S = dict(np.load(npz, allow_pickle=True))
    # reconstruct minimal ds list for figures
    ds = []
    n = len([k for k in S if k.startswith("J_")])
    for i in range(n):
        ds.append(dict(name=str(S["ds_names"][i]), channel=str(S["ds_channels"][i]), J=S[f"J_{i}"],
                       Jw2=S[f"Jw2_{i}"], sigma=S[f"sigma_{i}"], nbin=S[f"J_{i}"].shape[0],
                       edges=S[f"edges_{i}"], xlabel=str(S[f"xlabel_{i}"]), xscale=float(S[f"xscale_{i}"]),
                       nom=S[f"nom_{i}"], data=S[f"data_{i}"]))
    make_figures(ds, S, {})


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot_only(sys.argv[sys.argv.index("--plot-only") + 1])
    else:
        main()
