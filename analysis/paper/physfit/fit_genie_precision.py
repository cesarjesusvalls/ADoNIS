"""Fit ADoNIS (the existing 1.87M bank -- NO more generation) to the GENIE 3M predictions treated
AS DATA at GENIE's OWN precision, on a fixed 20-bin-per-observable binning.

Contrast with the earlier fits (logbook 9-14) that adopted the real T2K covariance (~10% bars) as
the error model, which HIDES the ADoNIS<->GENIE physics difference. Here the error model is GENIE's
OWN statistics plus a modelled systematic:

  error/bin = sqrt( GENIE_stat^2  +  (SYST * data)^2  [+ ADoNIS-bank MC^2] )

- GENIE_stat: Poisson error of the GENIE sample in the bin.
- SYST (default 5%): UNCORRELATED bin-by-bin systematic, added in quadrature (a modelled measurement
  systematic; diagonal covariance).
- ADoNIS-bank MC: the finite 1.87M bank's own per-bin MC error ("use the ADoNIS we have" -- fold it in
  honestly rather than pretend the model is exact). With 20 bins this is ~1%, subdominant to SYST;
  reported explicitly and toggle-able via ADONIS_INCLUDE_MC.

Binning: 20 equal-count (quantile) bins per observable -> ~uniform GENIE stat error; ADoNIS gets
~11k effective events/bin (~1% MC), no longer stat-limited (the 775-bin design was).

Absolute fit: both sides carry their physical normalisation (ADoNIS bank conv; GENIE per_event); the
known ~14% offset is absorbed by qe_norm/res_norm (blueprint: absolute by default).

UNITS: dsigma/dx is per reported unit -- GeV/c for dpt (load_cc0pi conv encodes the MeV->GeV factor;
the GENIE data divides by the bin width in GeV/c to match), rad for dat.

LIVE PROGRESS: per-iteration chi2 + knobs streamed with flush (python -u > log). Reuses
info_content._bin / bin_w0 (ADoNIS binning) + build_fakedata.extract_cc0pi (GENIE selection).
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from adonis.reweight import bank_plot as BP, bank_reweight as BR
from adonis.reweight.reweight_model import nominal_knobs
from analysis.paper import info_content as IC
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.paper.physfit.build_fakedata import extract_cc0pi

BANKDIR = os.environ.get("ADONIS_EVENT_BANK", "output/event_bank")
GST_MATCHED = os.environ.get("ADONIS_GST_MATCHED",
                             "output/altgen/genie_t2k_12C_ar23_CCQERES_3M.gst.root")
N_BINS = int(os.environ.get("ADONIS_NBINS", "20"))            # fixed bins per observable
SYST = float(os.environ.get("ADONIS_SYST", "0.05"))          # uncorrelated bin-by-bin systematic frac
INCLUDE_MC = os.environ.get("ADONIS_INCLUDE_MC", "1") == "1"  # fold in ADoNIS-bank MC error
BINMODE = os.environ.get("ADONIS_BINMODE", "uniform")        # "uniform" (over [0,p99]) or "quantile"
LABEL = os.environ.get("ADONIS_LABEL", "genieprec")
NIT = int(os.environ.get("ALTGEN_NIT", "15"))

FIT = ["M_A_qe", "qe_norm", "kF_sf", "s_NN_el", "sabs", "res_norm"]
LO = np.array([0.5, 0.3, 0.6, 0.3, 0.3, 0.3])
HI = np.array([2.0, 3.0, 1.6, 3.0, 3.0, 3.0])
NP = len(FIT)


def knobs_from(theta, nom):
    upd = {}
    for i, key in enumerate(FIT):
        if key == "s_NN_el":
            upd["s_NN_elastic"] = theta[i] * jnp.ones(3)
        else:
            upd[key] = theta[i]
    return nom._replace(**upd)


def design_edges(values, n_bins, mode="uniform", p_hi=99.0):
    """Bin edges for the GENIE observable.
    - "uniform": n_bins equal-width bins over [min, p_hi-percentile]; the sparse tail beyond p_hi is
      folded into the last (overflow) bin by clipping (applied identically to GENIE and ADoNIS), so the
      structure is resolved instead of one giant tail bin.
    - "quantile": equal-count bins (uniform GENIE stat error, but buries the tail)."""
    v = np.asarray(values)
    if mode == "quantile":
        edges = np.quantile(np.sort(v), np.linspace(0.0, 1.0, n_bins + 1))
    else:
        lo = float(v.min()); hi = float(np.percentile(v, p_hi))
        edges = np.linspace(lo, hi, n_bins + 1)
    edges = np.unique(edges)                        # guard against ties collapsing bins
    edges[0] -= 1e-6; edges[-1] += 1e-6             # inclusive extremes
    return edges


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    B = BP.load_bank(BANKDIR); JB = BR.to_jax(B); grids = BR.default_grids(); nom = nominal_knobs()
    log(f"bank {len(B['w0'])} events | GENIE gst {GST_MATCHED} | {N_BINS} bins/obs | "
        f"syst {SYST:.0%} | ADoNIS-MC {'on' if INCLUDE_MC else 'off'}")

    # ---- GENIE side (data): per-event CC0pi observables via the shared selection ----------------- #
    E = extract_cc0pi(GST_MATCHED)
    sel = E["sel"]; per_event = E["per_event"]
    gvals = {"dpt": E["dpt"][sel], "dat": E["dat"][sel]}
    log(f"GENIE CC0pi selected {int(sel.sum())} events (per_event={per_event:.4e} 1e-38/nucleon)")

    # ---- ADoNIS side (model): per-event observables + nominal weights for the MC-error floor ------ #
    lead0, _ = BP.leading_proton(B); sig0 = BP.signal_cc0pi(B)[0]
    avals = {"dpt": np.asarray(BP.dpt(B, lead0)), "dat": np.asarray(BP.dat(B, lead0))}
    w0 = np.asarray(BR.weight_jit(JB, nom, grids))          # nominal ADoNIS per-event weight
    log(f"ADoNIS CC0pi signal {int(sig0.sum())} events; nominal weights ready")

    # ---- build datasets: GENIE data (per reported unit) + GENIE-stat (+) syst (+) ADoNIS-MC cov --- #
    ds = []
    for obs, xlab, xsc in (("dpt", r"$\delta p_T$ [GeV/c]", 1000.0), ("dat", r"$\delta\alpha_T$ [rad]", 1.0)):
        edges = design_edges(gvals[obs], N_BINS, mode=BINMODE)  # MeV (dpt) / rad (dat)
        nb = len(edges) - 1
        _, conv, _, _ = IC.load_cc0pi(obs)                  # per-nucleon 1e-38 conv (encodes reported unit)
        bw = np.diff(edges)                                 # MeV (dpt) / rad (dat)
        bw_unit = bw / (1000.0 if obs == "dpt" else 1.0)    # dsigma per GeV/c (dpt) or per rad (dat)
        # fold the sparse tail into the last (overflow) bin by clipping STRICTLY INSIDE the range
        # (a value exactly on edges[0] would map to bin -1 via searchsorted); identical for GENIE/ADoNIS
        eps = (edges[-1] - edges[0]) * 1e-12
        gclip = np.clip(gvals[obs], edges[0] + eps, edges[-1] - eps)
        aclip = np.clip(avals[obs], edges[0] + eps, edges[-1] - eps)
        # GENIE data central + stat error (per reported unit -> matches the ADoNIS conv convention)
        cnt_g, _ = np.histogram(gclip, bins=edges)
        dsig_g = cnt_g * per_event / bw_unit
        stat_g = np.sqrt(cnt_g) * per_event / bw_unit
        syst_g = SYST * dsig_g                              # uncorrelated bin-by-bin systematic
        # ADoNIS binning (shared _bin) + nominal MC error per bin (per reported unit via conv)
        selA, bidx, nbA = IC._bin(sig0, aclip, edges)
        assert nbA == nb
        scale_bin = conv / bw
        sw = np.bincount(bidx, weights=w0[selA], minlength=nb)
        sw2 = np.bincount(bidx, weights=w0[selA]**2, minlength=nb)
        ado_val = scale_bin * sw
        ado_mcerr = scale_bin * np.sqrt(sw2)
        ado_effN = np.where(sw2 > 0, sw**2 / np.maximum(sw2, 1e-300), 0.0)
        # combined error model (quadrature, diagonal): GENIE stat (+) syst (+) ADoNIS MC
        var = stat_g**2 + syst_g**2 + (ado_mcerr**2 if INCLUDE_MC else 0.0)
        Cinv = np.diag(1.0 / np.maximum(var, 1e-300))
        ds.append(dict(name=f"CC0pi {obs}", key=obs, sel_idx=selA, binidx=bidx, nbin=nb,
                       scale_bin=scale_bin, offset=np.zeros(nb), data=dsig_g, Cinv=Cinv,
                       sigma=np.sqrt(var), edges=edges, xlabel=xlab, xscale=xsc,
                       stat_g=stat_g, syst_g=syst_g, ado_mcerr=ado_mcerr, ado_effN=ado_effN,
                       ado_val=ado_val, cnt_g=cnt_g))
        rel = lambda e: np.median(e / np.maximum(dsig_g, 1e-30))
        log(f"  [{obs}] {nb} bins | GENIE stat {rel(stat_g):.1%} | syst {SYST:.0%} | "
            f"ADoNIS MC {np.median(ado_mcerr/np.maximum(ado_val,1e-30)):.1%} (effN/bin {np.median(ado_effN):.0f}) "
            f"| TOTAL err/bin {rel(np.sqrt(var)):.1%}")

    nbins = sum(d["nbin"] for d in ds); ndf = nbins - NP
    log(f"total {nbins} bins, ndf={ndf}")

    # ---- differentiable model + Jacobian (info_content jvp pattern) ------------------------------ #
    def wf(theta, JB):
        return BR.bank_weight(JB, knobs_from(theta, nom), grids)
    wf_jit = jax.jit(wf)
    jvp_wf = jax.jit(lambda th, tang, JB: jax.jvp(lambda t: wf(t, JB), (th,), (tang,))[1])

    def model_bins(th):
        w = np.asarray(wf_jit(jnp.asarray(th), JB))
        return [IC.bin_w0(d, w) for d in ds]

    def jac_bins(th):
        Js = [np.zeros((d["nbin"], NP)) for d in ds]
        for k in range(NP):
            g = np.asarray(jvp_wf(jnp.asarray(th), jnp.zeros(NP).at[k].set(1.0), JB))
            for j, d in enumerate(ds):
                Js[j][:, k] = IC.bin_w0(d, g)
        return Js

    def chi2_of(mods):
        tot, per = 0.0, []
        for d, m in zip(ds, mods):
            r = m - d["data"]; c = float(r @ d["Cinv"] @ r); per.append(c); tot += c
        return tot, per

    th = np.ones(NP)
    mods = model_bins(th); c_nom, per_nom = chi2_of(mods)
    log(f"NOMINAL chi2/ndf = {c_nom/ndf:.2f}  (chi2={c_nom:.1f}, ndf={ndf})")
    for d, c in zip(ds, per_nom):
        log(f"    {d['name']:12s} chi2/nbin = {c/d['nbin']:.2f}")

    # ---- EARLY nominal-overlay figure (visible before the slow LM) ------------------------------- #
    figN, axN = plt.subplots(1, len(ds), figsize=(7.5 * len(ds), 5))
    for col, (d, mn) in enumerate(zip(ds, mods)):
        ctr = 0.5 * (d["edges"][1:] + d["edges"][:-1]) / d["xscale"]
        xerr = 0.5 * np.diff(d["edges"]) / d["xscale"]
        axN[col].errorbar(ctr, d["data"], xerr=xerr, yerr=d["sigma"], fmt="o", color="k", ms=4,
                          lw=0.8, capsize=0, label=f"GENIE 3M ({N_BINS} bins, stat+{SYST:.0%})")
        axN[col].plot(ctr, mn, color="C0", lw=1.6, label="ADoNIS nominal")
        axN[col].set(xlabel=d["xlabel"], ylabel=r"$d\sigma/dx$ [$10^{-38}$/nucleon]",
                     title=f"{d['name']}  (nominal chi2/nbin {per_nom[col]/d['nbin']:.1f})")
        axN[col].set_ylim(bottom=0)
        if col == 0:
            axN[col].legend(fontsize=9)
    figN.suptitle(f"GENIE-3M as data vs ADoNIS nominal  ({N_BINS} bins/obs, stat+{SYST:.0%} syst; "
                  f"pre-fit chi2/ndf {c_nom/ndf:.1f})")
    figN.tight_layout(); os.makedirs("output/figures", exist_ok=True)
    figN.savefig(f"output/figures/altgen_{LABEL}_nominal.png", dpi=140); plt.close(figN)
    log(f"[fig] output/figures/altgen_{LABEL}_nominal.png  (nominal overlay -- ready before the fit)")

    # ---- Levenberg-Marquardt with LIVE per-iteration progress ------------------------------------ #
    lam = 1e-3; c_cur = c_nom
    for it in range(NIT):
        c_before = c_cur
        Js = jac_bins(th); A = np.zeros((NP, NP)); g = np.zeros(NP)
        for d, m, J in zip(ds, mods, Js):
            r = m - d["data"]
            A += J.T @ d["Cinv"] @ J; g += J.T @ d["Cinv"] @ r
        gnorm = np.linalg.norm(g)
        for _ in range(12):
            dth = np.linalg.solve(A + lam * np.diag(np.maximum(np.diag(A), 1e-12)), -g)
            th_try = np.clip(th + dth, LO, HI)
            mods_try = model_bins(th_try); c_try, _ = chi2_of(mods_try)
            if c_try < c_cur:
                th, mods, c_cur = th_try, mods_try, c_try; lam = max(lam/3, 1e-8); break
            lam *= 5
        log(f"  it {it:3d}  chi2/ndf={c_cur/ndf:8.3f}  |grad|={gnorm:.2e} lam={lam:.1e}  " +
            " ".join(f"{k}={th[i]:.3f}" for i, k in enumerate(FIT)))
        if np.linalg.norm(dth) < 1e-6 or (c_before - c_cur) / max(c_before, 1e-9) < 1e-3:
            log(f"  converged/plateau (dchi2/chi2<1e-3 or step<1e-6) at it {it}"); break

    # ---- errors (Gauss-Newton Hessian; rails invalid) -------------------------------------------- #
    Js = jac_bins(th); A = np.zeros((NP, NP))
    for d, J in zip(ds, Js):
        A += J.T @ d["Cinv"] @ J
    V = np.linalg.pinv(A, rcond=1e-12); sig = np.sqrt(np.abs(np.diag(V)))
    railed = [(abs(th[i]-LO[i]) < 1e-3 or abs(th[i]-HI[i]) < 1e-3) for i in range(NP)]
    _, per_bf = chi2_of(mods)
    print(f"\n==== ADoNIS(1.87M) -> GENIE-3M as data  ({N_BINS} bins/obs, stat+{SYST:.0%} syst"
          f"{'+ADoNIS-MC' if INCLUDE_MC else ''};  chi2/ndf {c_nom/ndf:.2f} -> {c_cur/ndf:.2f}) ====")
    print(f"{'knob':>20} {'BFP':>8} {'+/-':>8} {'pull':>7} {'flag':>6}")
    for i, k in enumerate(FIT):
        print(f"{k:>20} {th[i]:8.3f} {sig[i]:8.3f} {(th[i]-1)/sig[i] if sig[i]>0 else 0:7.2f} "
              f"{'RAIL' if railed[i] else '':>6}")
    print("\nper-dataset chi2/nbin (nominal -> BFP):")
    for d, cn, cb in zip(ds, per_nom, per_bf):
        print(f"    {d['name']:12s} {cn/d['nbin']:7.2f} -> {cb/d['nbin']:7.2f}")

    # ---- figure: data vs nominal vs BFP + pulls, per observable ----------------------------------- #
    mods_nom = model_bins(np.ones(NP))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for col, d in enumerate(ds):
        ctr = 0.5 * (d["edges"][1:] + d["edges"][:-1]) / d["xscale"]
        xerr = 0.5 * np.diff(d["edges"]) / d["xscale"]
        ax = axes[0, col]
        ax.errorbar(ctr, d["data"], xerr=xerr, yerr=d["sigma"], fmt="o", color="k", ms=4, lw=0.8,
                    capsize=0, label=f"GENIE 3M ({N_BINS} bins, stat+{SYST:.0%})")
        ax.plot(ctr, mods_nom[col], color="0.55", ls=":", lw=1.6, label="ADoNIS nominal")
        ax.plot(ctr, mods[col], color="C0", lw=1.8, label="ADoNIS BFP")
        ax.set(xlabel=d["xlabel"], ylabel=r"$d\sigma/dx$ [$10^{-38}$/nucleon]", title=d["name"])
        ax.set_ylim(bottom=0)
        if col == 0:
            ax.legend(fontsize=9)
        axp = axes[1, col]
        pull = (mods[col] - d["data"]) / d["sigma"]
        axp.bar(ctr, pull, width=2*xerr*0.9, color="C3", alpha=0.7)
        axp.axhline(0, color="k", lw=0.8); axp.axhline(1, color="0.7", lw=0.6, ls="--"); axp.axhline(-1, color="0.7", lw=0.6, ls="--")
        axp.set(xlabel=d["xlabel"], ylabel=r"(BFP$-$data)/$\sigma$",
                title=f"post-fit pull  (chi2/nbin {per_bf[col]/d['nbin']:.2f})")
    fig.suptitle(f"ADoNIS(1.87M bank) fit to GENIE-3M as data  ({N_BINS} bins/obs, stat+{SYST:.0%} syst"
                 f"{'+ADoNIS-MC' if INCLUDE_MC else ''};  chi2/ndf {c_nom/ndf:.2f} -> {c_cur/ndf:.2f})")
    fig.tight_layout()
    out = f"output/figures/altgen_{LABEL}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    np.savez(f"output/altgen/{LABEL}.npz", fit=FIT, bfp=th, sig=sig, V=V,
             chi2_nom=c_nom, chi2_bf=c_cur, nbins=nbins, ndf=ndf, nbins_per_obs=N_BINS, syst=SYST,
             include_mc=INCLUDE_MC, per_nom=per_nom, per_bf=per_bf,
             **{f"{d['key']}_edges": d["edges"] for d in ds},
             **{f"{d['key']}_data": d["data"] for d in ds},
             **{f"{d['key']}_sigma": d["sigma"] for d in ds},
             **{f"{d['key']}_bfp": mods[i] for i, d in enumerate(ds)},
             **{f"{d['key']}_nom": mods_nom[i] for i, d in enumerate(ds)},
             **{f"{d['key']}_stat_g": d["stat_g"] for d in ds},
             **{f"{d['key']}_syst_g": d["syst_g"] for d in ds},
             **{f"{d['key']}_ado_mcerr": d["ado_mcerr"] for d in ds},
             **{f"{d['key']}_ado_effN": d["ado_effN"] for d in ds})
    log(f"[fig] {out}")
    log("done")


if __name__ == "__main__":
    main()
