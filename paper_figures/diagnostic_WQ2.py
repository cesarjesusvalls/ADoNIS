"""CANONICAL RES/QE diagnostic: dsigma/dW(or omega) & dsigma/dQ^2, ADoNIS vs the REAL ACHILLES
hepmc events, with ratio panels.  This is the standing diagnostic for the RES low-Q^2 deficit
investigation (res_investigation_log.md; doc retired to git history).  Regenerate after every candidate fix.

ACHILLES reference: unweighted events from a NuHepMC hepmc.  sigma_total read from the final
GenCrossSection (pb -> nb).  Q^2 = -(k_nu - k_mu)^2 with k_nu = MAX-ENERGY pid14 (the status-2
beam, NOT the E=30 status-4 placeholder); k_mu = pid13 status1.  W = M(Npi) = M(p_N + p_pi).

Usage: python paper_figures/diagnostic_WQ2.py res 2000000
       python paper_figures/diagnostic_WQ2.py qe  2000000
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax; jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
ACH = ROOT.parent / "Achilles/_resrun_out"


def parse_hepmc(path, mode):
    """Return (left[MeV or MeV], Q2[GeV^2], w[nb]) per event."""
    left, q2, raw = [], [], []
    nu, mu, pN, pPi, Wline, gx = [], None, None, None, None, [None]
    def mink2(p): return p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2
    def flush():
        nonlocal nu, mu, pN, pPi, Wline
        if nu and mu is not None:
            knu = max(nu, key=lambda p: p[0]); q = knu - mu
            q2.append(-mink2(q) / 1e6)
            if mode == "res" and pN is not None and pPi is not None:
                left.append(np.sqrt(max(mink2(pN + pPi), 0.0)))
            else:  # qe: omega = Enu - Emu
                left.append(knu[0] - mu[0])
            raw.append(Wline if Wline is not None else 1.0)
        nu, mu, pN, pPi, Wline = [], None, None, None, None
    with open(path) as fh:
        for line in fh:
            t = line[:2]
            if t == "E ":
                flush()
            elif t == "W ":
                try: Wline = float(line.split()[1])
                except ValueError: pass            # skip the "W CV" header line
            elif t[:1] == "A" and "GenCrossSection" in line:
                gx[0] = float(line.split()[3])      # pb, cumulative (last wins)
            elif t == "P ":
                f = line.split(); pid = int(f[3]); st = int(f[9])
                p4 = np.array([float(f[7]), float(f[4]), float(f[5]), float(f[6])])
                if pid == 14: nu.append(p4)
                elif pid == 13 and st == 1: mu = p4
                elif st == 1 and abs(pid) in (2112, 2212): pN = p4
                elif st == 1 and pid in (211, 111, -211): pPi = p4
    flush()
    left = np.array(left); q2 = np.array(q2); raw = np.array(raw, float)
    sigma_nb = gx[0] * 1e-3                              # pb -> nb
    w = raw / raw.sum() * sigma_nb
    return left, q2, w, sigma_nb


def adonis(mode, n, nchunks=25):
    """Run the ADoNIS generator in `nchunks` chunks (one seed each) with % progress.
    Each chunk's per-event weights are normalised to sum to that chunk's sigma estimate, then
    divided by nchunks, so the concatenated set sums to the MEAN sigma over chunks."""
    import time
    chunk = max(n // nchunks, 1)
    knu_l, kmu_l, w_l, left_l, sig_chunks = [], [], [], [], []
    t0 = time.time()
    for i in range(nchunks):
        if mode == "res":
            from adonis.xsec import res_xsec as M
            r = M.generate(chunk, seed=i, return_events=True); ev = r["events"]
            knu, kmu = ev["k_nu"], ev["k_mu"]
            wc = ev["w"]                                   # already sums to sigma_chunk
            left = np.sqrt(np.clip((ev["p_N"][:, 0] + ev["p_pi"][:, 0])**2
                           - np.sum((ev["p_N"][:, 1:] + ev["p_pi"][:, 1:])**2, axis=1), 0, None))
        else:
            from adonis.xsec import qe_xsec as M
            o = M.sample_importance(chunk, seed=i)
            knu, kmu = o["k_nu"], o["k_mu"]
            wc = o["w"] / chunk                            # sample_importance: sigma = mean -> /chunk
            left = knu[:, 0] - kmu[:, 0]
        sig_chunks.append(float(wc.sum()))
        knu_l.append(knu); kmu_l.append(kmu); w_l.append(wc / nchunks); left_l.append(left)
        run = np.mean(sig_chunks)
        print(f"  [ADoNIS {mode}] {100*(i+1)/nchunks:5.1f}%  ({(i+1)*chunk:,}/{nchunks*chunk:,} ev)"
              f"  running σ={run:.4e}  {time.time()-t0:.0f}s", flush=True)
    knu = np.concatenate(knu_l); kmu = np.concatenate(kmu_l)
    w = np.concatenate(w_l); left = np.concatenate(left_l)
    q = knu - kmu
    q2 = (np.sum(q[:, 1:]**2, axis=1) - q[:, 0]**2) / 1e6
    keep = w > 0
    return left[keep], q2[keep], w[keep]


def panel(ax, axr, ach_x, ach_w, ado_x, ado_w, bins, xlabel, axc=None):
    bw = np.diff(bins); ctr = 0.5 * (bins[1:] + bins[:-1])
    ha, _ = np.histogram(ach_x, bins=bins, weights=ach_w)
    h2a, _ = np.histogram(ach_x, bins=bins, weights=ach_w**2)
    hd, _ = np.histogram(ado_x, bins=bins, weights=ado_w)
    h2d, _ = np.histogram(ado_x, bins=bins, weights=ado_w**2)
    da, dd = ha / bw, hd / bw
    ea, ed = np.sqrt(h2a) / bw, np.sqrt(h2d) / bw
    ax.step(bins, np.append(da, da[-1]), where="post", color="0.35", label="ACHILLES")
    ax.errorbar(ctr, da, yerr=ea, fmt="none", ecolor="0.35", alpha=0.6)
    ax.step(bins, np.append(dd, dd[-1]), where="post", color="C3", label="ADoNIS")
    ax.errorbar(ctr, dd, yerr=ed, fmt="none", ecolor="C3", alpha=0.6)
    ax.legend(); ax.set_ylabel("dσ/d" + xlabel)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = dd / da
        re = r * np.sqrt((ed / dd)**2 + (ea / da)**2)
    axr.axhspan(0.97, 1.03, color="green", alpha=0.15); axr.axhline(1.0, ls="--", color="green")
    axr.step(bins, np.append(r, r[-1]), where="post", color="C3", lw=1.3)   # ratio as steps
    axr.errorbar(ctr, r, yerr=re, fmt="none", ecolor="C3", alpha=0.8, capsize=1.5)
    axr.set_ylim(0.4, 1.2); axr.set_ylabel("ADoNIS/ACH")
    if axc is None:
        axr.set_xlabel(xlabel)
    # chi2/ndf between the two histograms (propagated stat errors), over bins both populated
    msk = (da > 0) & (dd > 0) & np.isfinite(ed) & np.isfinite(ea) & ((ed**2 + ea**2) > 0)
    ndf = int(msk.sum())
    chi2_bins = np.where(msk, (dd - da)**2 / np.where(msk, ed**2 + ea**2, 1.0), 0.0)
    chi2 = float(chi2_bins[msk].sum())
    chi2ndf = chi2 / max(ndf, 1)
    axr.text(0.03, 0.82, f"χ²/ndf = {chi2ndf:.2f}  ({ndf} bins)", transform=axr.transAxes, fontsize=9)
    # per-bin chi2 contribution (disaggregated) -> spot which bins drive the disagreement
    if axc is not None:
        axc.bar(ctr, chi2_bins, width=bw, color="C3", alpha=0.6, align="center")
        axc.axhline(1.0, ls=":", color="0.5", lw=0.8)          # chi2/bin = 1 reference
        axc.set_ylabel("χ²/bin"); axc.set_xlabel(xlabel); axc.set_ylim(bottom=0)
    return chi2ndf


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "res"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000000
    hepmc = ACH / ("nofsi_res_hi.hepmc" if mode == "res" else "nofsi_qe_hi.hepmc")
    if not hepmc.exists():
        hepmc = ACH / ("nofsi_res.hepmc" if mode == "res" else "nofsi_qe.hepmc")
    print(f"[{mode}] ACHILLES ref: {hepmc.name}")
    alx, aq2, aw, sig_ach = parse_hepmc(hepmc, mode)
    print(f"  {len(alx)} ACHILLES events, sigma_total={sig_ach:.4e} nb")
    print(f"[{mode}] ADoNIS n={n} ...")
    dlx, dq2, dw = adonis(mode, n)
    sig_ado = dw.sum()
    print(f"  ADoNIS sigma={sig_ado:.4e} nb  total ADoNIS/ACH={sig_ado/sig_ach:.3f}")

    if mode == "res":
        lbins = np.linspace(1080, 2000, 16); llab = "W=M(Nπ) [MeV]"
    else:
        lbins = np.linspace(0, 2000, 22); llab = "ω=E_ν−E_μ [MeV]"
    qbins = np.linspace(0, 2.0, 18)
    fig, axes = plt.subplots(3, 2, figsize=(14, 9.5), height_ratios=[3, 1, 1], sharex="col")
    c2W = panel(axes[0, 0], axes[1, 0], alx, aw, dlx, dw, lbins, llab, axc=axes[2, 0])
    c2Q = panel(axes[0, 1], axes[1, 1], aq2, aw, dq2, dw, qbins, "Q² [GeV²]", axc=axes[2, 1])
    fig.suptitle(f"{mode.upper()} primary (no FSI): ACHILLES vs ADoNIS — total ADoNIS/ACH = "
                 f"{sig_ado/sig_ach:.3f}   (N_ACH={len(alx)}, N_ADO={len(dlx)})   "
                 f"χ²/ndf  W={c2W:.2f}  Q²={c2Q:.2f}")
    fig.tight_layout()
    out = ROOT / "paper_figures" / (f"res_WQ2_shapes.png" if mode == "res" else "qe_WQ2_shapes_5x.png")
    fig.savefig(out, dpi=110); print("  wrote", out)
    # low/high Q2 integrated ratio
    for cut in (0.2, 0.4):
        la = aw[aq2 < cut].sum(); ld = dw[dq2 < cut].sum()
        ha = aw[aq2 >= cut].sum(); hd = dw[dq2 >= cut].sum()
        print(f"  Q2<{cut}: ADO/ACH={ld/la:.3f}   Q2>={cut}: ADO/ACH={hd/ha:.3f}")


if __name__ == "__main__":
    main()
