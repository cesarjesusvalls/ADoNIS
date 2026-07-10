"""Build the PHYSICAL-FIT report: 4 figures (PNG) + a self-contained PDF (logbook 17).

Reads the saved run npz files (gate1/closure/closure_fluct/inject2x/inject2x_v2/genie), rebuilds
the engine to evaluate model curves at the saved BFPs, and reproduces excised regions with the SAME
flags() used by the fits (imported from physical_fit_run — no re-implementation).

Palette (validated, CVD-safe; identity never color-alone -- distinct markers + direct labels):
  data=black points | ADoNIS nominal=gray dotted | M0=red squares | M1=blue circles | M2=orange triangles
"""
import os, sys, time, textwrap
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.t2k.differentiability import bank_plot as BP, bank_reweight as BR
from analysis.t2k.differentiability.full_knobs import nominal_knobs
from physical_fit import PNAMES, PRIOR, NPAR, build_physfit_datasets, theta_nominal, SYST
from physical_fit_run import Engine, apply_mode, flags

C_M0, C_M1, C_M2, C_NOM = "#d62728", "#1f77b4", "#ff7f0e", "0.45"
OUTFIG = "output/figures"; OUTREP = "output/reports"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def load(name):
    return np.load(f"output/altgen/{name}.npz", allow_pickle=True)


def _step_xy(edges, y):
    return np.repeat(edges, 2)[1:-1], np.repeat(y, 2)


def stairs_band(ax, edges, y, rel, color, label=None, ls="-", lw=1.5, alpha=1.0, band=True, balpha=0.16):
    """Histogram-step model curve over the bin edges + ADoNIS-MC relative-error band."""
    xs, ys = _step_xy(edges, y)
    ax.plot(xs, ys, color=color, ls=ls, lw=lw, alpha=alpha, label=label, zorder=2)
    if band:
        ax.fill_between(xs, np.repeat(y * (1 - rel), 2), np.repeat(y * (1 + rel), 2),
                        color=color, alpha=balpha, lw=0, zorder=1)


def snapshot(ds):
    return [{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()} for d in ds]


def restore(ds, snap):
    for d, s in zip(ds, snap):
        d.clear(); d.update({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()})
        d.pop("stat_g", None) if "stat_g" not in s else None


def main():
    g1 = load("physfit_gate1")
    clo = load("physfit_closure"); clf = load("physfit_closure_fluct")
    i2 = load("physfit_inject2x"); i2v = load("physfit_inject2x_v2")
    ge = load("physfit_genie")
    os.makedirs(OUTFIG, exist_ok=True); os.makedirs(OUTREP, exist_ok=True)

    # ---- engine + model curves (model depends on theta only, not on the data mode) ---------------- #
    B = BP.load_bank(os.environ.get("ADONIS_EVENT_BANK", "output/event_bank"))
    JB = BR.to_jax(B); grids = BR.default_grids(); nom = nominal_knobs()
    w0 = np.asarray(BR.weight_jit(JB, nom, grids))
    ds = build_physfit_datasets(B, w0, log)
    eng = Engine(ds, JB, grids, nom)
    th0 = theta_nominal(nom)
    log("evaluating model curves at saved BFPs")
    m_nom = eng.model(th0)
    m_M0_2x = eng.model(np.asarray(i2["M0_th"]))
    m_M2_2x = eng.model(np.asarray(i2["M2_th"]))
    m_M0_ge = eng.model(np.asarray(ge["M0_th"]))
    m_M1_ge = eng.model(np.asarray(ge["M1_th"]))
    m_M2_ge = eng.model(np.asarray(ge["M2_th"]))
    # per-bin RELATIVE bank-MC error (same reweighted events -> same relative error for every curve)
    relerr = [d["mcerr"] / np.maximum(seg_ := m_nom[eng.row0[j]:eng.row0[j+1]], 1e-300)
              for j, d in enumerate(ds)]
    snap = snapshot(ds)
    row0 = eng.row0

    def seg(m, j):
        return m[row0[j]:row0[j+1]]

    # ================= FIG 1: Gate I ================================================================ #
    shrink = np.asarray(g1["shrink"]); names = list(g1["pnames"])
    order = np.argsort(shrink)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    cols = [C_M1 if shrink[k] < 0.5 else "0.75" for k in order]
    ax.barh(range(NPAR), shrink[order], color=cols, height=0.72)
    ax.set_yticks(range(NPAR)); ax.set_yticklabels([names[k] for k in order], fontsize=8)
    ax.axvline(0.5, color="k", lw=1.0, ls="--")
    ax.text(0.51, 6.5, "Gate I threshold\n(shrinkage = 0.5)", fontsize=8, va="top")
    ax.set_xlabel(r"posterior shrinkage  $\sigma_{\rm post}/\sigma_{\rm prior}$")
    ax.set_title("Gate I — which knobs can these samples inform?  (Asimov Fisher, 20% priors)")
    ax.set_xlim(0, 1.05); ax.invert_yaxis(); ax.grid(axis="x", color="0.92", lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUTFIG}/physfit_fig1_gate1.png", dpi=150)
    log("[fig1] gate1")

    # ================= FIG 2: closure (noiseless + fluctuated) ===================================== #
    sub = [int(k) for k in clo["subset"]]; nsub = len(sub)
    truth_c = np.asarray(clo["truth"])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    yy = np.arange(nsub)
    for dyi, (npz_, lab, col, mk) in enumerate([
            (clo, "noiseless closure (M0=M1=M2)", C_M1, "o"),
            (clf, "fluctuated closure (M1, seed 20260710)", C_M2, "^")]):
        th = np.asarray(npz_["M1_th"]); V = np.asarray(npz_["M1_V"])
        sig = np.sqrt(np.abs(np.diag(V)))
        pulls = [(th[k] - truth_c[i_]) / sig[c] for c, (k, i_) in enumerate(zip(sub, sub))]
        pulls = [(th[k] - truth_c[k]) / sig[c] for c, k in enumerate(sub)]
        ax.errorbar(pulls, yy + (dyi - 0.5) * 0.22, xerr=1.0, fmt=mk, color=col, ms=6,
                    capsize=3, lw=1.4, label=lab)
    ax.axvline(0, color="k", lw=1.0)
    ax.axvspan(-2, 2, color="0.94", zorder=0)
    ax.set_yticks(yy); ax.set_yticklabels([PNAMES[k] for k in sub])
    ax.set_xlabel(r"(BFP $-$ truth) / $\sigma$"); ax.set_xlim(-4, 4)
    ax.set_title("Step 2 — closure: injected kF_sf=1.10, M_A_res=0.85 recovered blind", fontsize=11)
    ax.legend(fontsize=8, loc="lower right"); ax.invert_yaxis()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUTFIG}/physfit_fig2_closure.png", dpi=150)
    log("[fig2] closure")

    # ================= FIG 3: inject2x ============================================================== #
    truth2, _ = apply_mode(ds, eng, "inject2x")   # rebuild the injected data in-place (same code path)
    jdpt = next(j for j, d in enumerate(ds) if d["key"] == "dpt")
    d = ds[jdpt]; x = 0.5 * (d["edges"][1:] + d["edges"][:-1]); xe = 0.5 * np.diff(d["edges"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    a1.axvspan(300, d["edges"][-1], color="#f5e6c8", zorder=0)
    a1.text(0.985, 0.97, "injected ×2 region\n(excised by M1)", transform=a1.transAxes,
            ha="right", va="top", fontsize=8, color="0.35")
    a1.errorbar(x, d["data"], xerr=xe, yerr=d["sigma"], fmt="o", color="k", ms=3.5, lw=0.8,
                capsize=0, label="fake data (Asimov, ×2 above 300 MeV)", zorder=3)
    stairs_band(a1, d["edges"], seg(m_nom, jdpt), relerr[jdpt], C_NOM, ls=":",
                label="ADoNIS truth/nominal (= M1 result)")
    stairs_band(a1, d["edges"], seg(m_M0_2x, jdpt), relerr[jdpt], C_M0, label="M0 traditional best fit")
    stairs_band(a1, d["edges"], seg(m_M2_2x, jdpt), relerr[jdpt], C_M2, label="M2 Huber best fit")
    a1.set(xlabel=r"$\delta p_T$ [MeV/c]", ylabel=r"$d\sigma/dx$ [$10^{-38}$/nucleon]",
           title="Step 3 — the artifact drags the traditional fit")
    a1.set_ylim(bottom=0); a1.legend(fontsize=8)
    # (b) per-method knob pulls vs TRUTH
    sub2 = [int(k) for k in i2["subset"]]
    yy = np.arange(len(sub2))
    for npz_, key, lab, col, mk, dy in [
            (i2, "M0", "M0 traditional", C_M0, "s", -0.25),
            (i2, "M2", "M2 Huber robust", C_M2, "^", 0.0),
            (i2v, "M1", "M1 physical (excise-refit)", C_M1, "o", 0.25)]:
        th = np.asarray(npz_[f"{key}_th"]); V = np.asarray(npz_[f"{key}_V"])
        sig = np.sqrt(np.abs(np.diag(V)))
        pulls = np.array([(th[k] - truth2[k]) / max(sig[c], 1e-12) for c, k in enumerate(sub2)])
        shown = np.clip(pulls, -14.5, 14.5)
        a2.scatter(shown, yy + dy, color=col, marker=mk, s=34, label=lab, zorder=3)
        for c, (p, s_) in enumerate(zip(pulls, shown)):
            if abs(p) > 14.5:
                a2.annotate(f"{p:+.0f}σ", (s_, yy[c] + dy), textcoords="offset points",
                            xytext=(-4 if p < 0 else 4, -3), fontsize=7, color=col,
                            ha="right" if p < 0 else "left")
    a2.axvline(0, color="k", lw=1.0); a2.axvspan(-2, 2, color="0.94", zorder=0)
    a2.set_yticks(yy); a2.set_yticklabels([PNAMES[k] for k in sub2], fontsize=8)
    a2.set_xlabel(r"(BFP $-$ truth) / $\sigma$"); a2.set_xlim(-15.5, 15.5)
    a2.set_title("parameter bias by method")
    a2.legend(fontsize=8, loc="lower left"); a2.invert_yaxis()
    for ax_ in (a1, a2):
        for s in ("top", "right"):
            ax_.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUTFIG}/physfit_fig3_inject2x.png", dpi=150)
    log("[fig3] inject2x")

    # ================= FIG 4: GENIE ================================================================ #
    restore(ds, snap)
    _, desc = apply_mode(ds, eng, "genie")
    fl_nom = flags(eng, m_nom)                      # the regions M1 excised (computed at nominal)
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3)
    axs = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(5)]
    for j, (ax, d) in enumerate(zip(axs, ds)):
        x = 0.5 * (d["edges"][1:] + d["edges"][:-1]); xe = 0.5 * np.diff(d["edges"])
        clean = np.ones(d["nbin"], bool)
        for f in [f_ for f_ in fl_nom if f_["j"] == j]:
            ax.axvspan(d["edges"][f["i0"]], d["edges"][f["i1"] + 1], color="#f0e0e0", zorder=0)
            clean[f["i0"]:f["i1"] + 1] = False
        ax.errorbar(x, d["data"], xerr=xe, yerr=d["sigma"], fmt="o", color="k", ms=2.8, lw=0.7,
                    capsize=0, zorder=3, label="GENIE 3M data")
        stairs_band(ax, d["edges"], seg(m_nom, j), relerr[j], C_NOM, ls=":", lw=1.3,
                    label="ADoNIS nominal", balpha=0.12)
        stairs_band(ax, d["edges"], seg(m_M2_ge, j), relerr[j], C_M2, lw=1.3,
                    label="M2 Huber fit", balpha=0.12)
        # M1: dashed/faint everywhere (extrapolation), solid overlay on the CLEAN (fitted) runs
        stairs_band(ax, d["edges"], seg(m_M1_ge, j), relerr[j], C_M1, ls="--", lw=1.2, alpha=0.5,
                    label="M1 extrapolation (excised)", balpha=0.10)
        m1j = seg(m_M1_ge, j); lab = "M1 clean-region fit"
        i = 0
        while i < d["nbin"]:
            if clean[i]:
                k = i
                while k + 1 < d["nbin"] and clean[k + 1]:
                    k += 1
                stairs_band(ax, d["edges"][i:k + 2], m1j[i:k + 1], relerr[j][i:k + 1], C_M1,
                            lw=1.8, label=lab, band=False)
                lab = None
                i = k + 1
            else:
                i += 1
        XLAB = {"dpt": r"$\delta p_T$ [MeV/c]", "dat": r"$\delta\alpha_T$ [rad]",
                "pn": r"$p_N$ [MeV/c]", "dptt": r"$\delta p_{TT}$ [MeV/c]",
                "daT": r"$\delta\alpha_T$ [deg]"}
        ax.set_title(d["name"], fontsize=10); ax.set_ylim(bottom=0)
        ax.set_xlabel(XLAB[d["key"]], fontsize=9)
        if j == 0:
            ax.legend(fontsize=6.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].text(0.03, 0.05, "shaded = excised\n(unknown-unknown\ncandidates)", fontsize=7.5,
                transform=axs[0].transAxes, color="0.35")
    # (f) knob comparison M0 vs M1, in prior units around nominal
    axk = fig.add_subplot(gs[1, 2])
    subg = [int(k) for k in ge["subset"]]
    yy = np.arange(len(subg))
    for key, lab, col, mk, dy in [("M0", "M0 traditional", C_M0, "s", -0.22),
                                  ("M2", "M2 Huber", C_M2, "^", 0.0),
                                  ("M1", "M1 physical", C_M1, "o", 0.22)]:
        th = np.asarray(ge[f"{key}_th"]); V = np.asarray(ge[f"{key}_V"])
        sig = np.sqrt(np.abs(np.diag(V)))
        val = np.array([(th[k] - th0[k]) / PRIOR[k] for k in subg])
        err = np.array([sig[c] / PRIOR[k] for c, k in enumerate(subg)])
        axk.errorbar(val, yy + dy, xerr=err, fmt=mk, color=col, ms=5, capsize=3, lw=1.3, label=lab)
    axk.axvline(0, color="k", lw=1.0)
    axk.set_yticks(yy); axk.set_yticklabels([PNAMES[k] for k in subg], fontsize=8)
    axk.set_xlabel(r"(BFP $-$ nominal) / $\sigma_{\rm prior}$")
    axk.set_title("knob pulls: all M0 pulls are\nQ-incoherent; M1's are coherent", fontsize=9)
    axk.legend(fontsize=7.5); axk.invert_yaxis()
    for s in ("top", "right"):
        axk.spines[s].set_visible(False)
    fig.suptitle("Step 4 — GENIE-3M as data: measure the coherent region, excise the rest", y=0.98)
    fig.savefig(f"{OUTFIG}/physfit_fig4_genie.png", dpi=150, bbox_inches="tight")
    log("[fig4] genie")

    # ================= PDF ========================================================================= #
    def textpage(pdf, title, body, fs=9.5):
        f = plt.figure(figsize=(8.27, 11.69))
        f.text(0.08, 0.93, textwrap.fill(title, 58), fontsize=14, weight="bold", va="bottom")
        f.text(0.08, 0.90, body, fontsize=fs, va="top", family="DejaVu Sans", linespacing=1.45)
        pdf.savefig(f); plt.close(f)

    def figpage(pdf, png, caption):
        img = plt.imread(png)
        f = plt.figure(figsize=(8.27, 11.69))
        ar = img.shape[0] / img.shape[1]
        w = 0.88; h = w * ar * (8.27 / 11.69)
        axi = f.add_axes([0.06, 0.86 - h, w, h]); axi.imshow(img); axi.axis("off")
        f.text(0.08, 0.83 - h, caption, fontsize=8.5, va="top", wrap=True, linespacing=1.4)
        pdf.savefig(f); plt.close(f)

    W = lambda s: "\n".join(textwrap.fill(p, 100) if not p.startswith(" ") else p
                            for p in s.strip().split("\n"))
    pdf_path = f"{OUTREP}/physical_fit_report.pdf"
    with PdfPages(pdf_path) as pdf:
        textpage(pdf, "The Physical Fit: unbiased inference in the presence of unknown unknowns", W(f"""
Session Callum_C, branch callum_c_altgen — 2026-07-10.  Evidence: docs/logbook/altgen_fakedata.md §17.

PROBLEM.  Every neutrino-interaction generator is approximate: uncertain parameters (tunable),
computational simplifications, and unknown unknowns (e.g. 2p2h before it was modeled). A standard
tune minimizes a global chi2 and will happily corrupt its physics parameters to absorb mismodeling
it cannot represent — producing confident-looking, unphysical "measurements". The question of this
study: how can we learn from data without being biased by mismodeling and unknown unknowns?

PRINCIPLE.  A parameter variation is a measurement only if the bins that parameter touches demand it
COHERENTLY. If half its responsive bins pull it up and half pull it down — with the net set by a
mismodeled tail — the fitted value is not physics. The physical fit (M1) enforces this with three
mechanisms, all calibrated statistics rather than heuristics:

  Gate I   (informativeness)  Asimov Fisher with priors: fit only knobs the sample can actually
           inform (posterior shrinkage < 0.5); freeze the rest.
  Gate II  (coherence)  Per-knob Cochran's Q heterogeneity test on the per-bin demand estimates
           delta-theta_kb = r_b/J_kb, plus a vector split-fit Q_split (low/high kinematic halves)
           that catches cross-knob compensation. Incoherently-pulled knobs are frozen.
  Excise-refit  Contiguous >2 sigma residual regions are excised as unknown-unknown CANDIDATES and
           the gated fit is re-run on the clean region — recovering real measurements from the
           coherent bins plus a localized map of what the model cannot describe.

Comparators: M0 = traditional global-chi2 fit (control). M2 = Huber robust loss (bounded per-bin
influence — the principled "penalize in the loss" alternative).

RESULT (one line): on data containing a known artifact, M0 biases parameters by up to 27 sigma;
M1 returns exact truth AND localizes the artifact; on real foreign-generator (GENIE) data, M1
converts four incoherent M0 pulls into one coherent physics measurement plus an excision map.

CONFIG.  ADoNIS 1.87M-event differentiable bank; 5 observables (T2K CC0pi dpt/dat + CC1pi
pN/dpTT/daT), 20 uniform bins each (p99 + overflow fold); diagonal errors sigma^2 = (5% syst)^2 +
ADoNIS-MC^2 (+ GENIE stat for step 4); 20% uncorrelated priors (Eb_shift ±4 MeV, f_NN_cex ±0.1).

SCOREBOARD.
  Step 1  Gate I           5/27 knobs informable: Eb_shift, kF_sf, s_NN_el[pn], f_NN_cex, M_A_res
  Step 2  closure          exact blind recovery, zero false freezes/flags (also with 1-sigma noise)
  Step 3  x2 artifact      M0 biased to -27 sigma | M1 exact truth + artifact excised | M2 partial
  Step 4  GENIE as data    M0 4 confident incoherent pulls | M1 coherent M_A_res = 0.868 +- 0.039
                           + 51/100 bins excised (dpt shape, dat norm offset, CC1pi structures)
"""))
        figpage(pdf, f"{OUTFIG}/physfit_fig1_gate1.png", W("""
FIG 1 — Gate I (Asimov Fisher, 100 bins, 5% syst, 20% priors). Posterior shrinkage per knob; only
5/27 pass the 0.5 threshold (blue). Notably, M_A_qe / qe_norm / res_norm / axial_strength / sf_norm
freeze NOT from insensitivity but from mutual degeneracy — the marginalized posterior punishes each
member of a collinear group. These are exactly the knobs earlier unphysical fits railed; Gate I
removes that freedom a priori. Known v3 refinement: fit one eigen-combination representative per
degenerate group (see Fig 4 for why)."""))
        figpage(pdf, f"{OUTFIG}/physfit_fig2_closure.png", W("""
FIG 2 — Step 2, closure. kF_sf = 1.10 and M_A_res = 0.85 injected via the exact reweight (blind to
the fit): all methods recover truth exactly on noiseless Asimov (chi2_data = 0; residual chi2 = the
prior penalty of the injected truth, exactly (0.5 sigma)^2 + (0.75 sigma)^2). With 1-sigma data
fluctuations (orange), M1 shows zero false freezes (Cochran-Q p = 0.92-0.96), zero false flags,
Q_split p = 0.23 — the gates stay quiet on healthy noisy data (single-seed null calibration)."""))
        figpage(pdf, f"{OUTFIG}/physfit_fig3_inject2x.png", W("""
FIG 3 — Step 3, the decisive controlled test. Data = ADoNIS Asimov with all dpt bins above 300 MeV
multiplied by 2 (an injected "unknown unknown"; truth = all knobs nominal). Model curves are
histogram steps over the analysis bins; shaded bands = the ADoNIS bank's own per-bin MC-stat
uncertainty. Left: the traditional fit (red) is dragged by the artifact across the whole spectrum;
M2 (orange) partially resists. Right: parameter biases — M0
corrupts every knob (kF_sf -11 sigma, s_NN_el[pn] -27 sigma, Eb_shift +8 sigma); M2 (Huber) reduces
but does not eliminate the bias; M1 refuses every incoherent pull (all Cochran-Q fail, Q_split
p = 3e-12), excises exactly the 12 injected bins, refits the clean region, and lands on truth to
0.00 sigma with real uncertainties. Same data, three answers — only one of them is physics."""))
        figpage(pdf, f"{OUTFIG}/physfit_fig4_genie.png", W("""
FIG 4 — Step 4, genuine foreign physics: GENIE 3.04 (AR23_20i, QE+RES, 3M events, T2K-numu/12C) as
data at its own statistical precision (+5% syst). Model curves are histogram steps with the ADoNIS
bank MC-stat band. Shaded = the 51/100 bins M1 excised at nominal: most of dpt (the
initial-state/FSI shape difference), ALL of dat (the ~20% normalization offset), and four CC1pi
structures. M1 is drawn SOLID where it was fitted (clean region) and DASHED where it merely
extrapolates into excised regions — the residual there is the flagged mismodeling, deliberately
NOT chased by the fit (M0/M2 do chase it, corrupting their parameters).
On the coherent remainder M1 measures M_A_res = 0.868 +- 0.039 — a
genuine, coherent RES-shape difference (GENIE Berger-Sehgal softer than ADoNIS DCC) — with kF_sf =
1.031 +- 0.026, s_NN_el[pn] = 0.939 +- 0.100, f_NN_cex = 0.514 +- 0.057. Bottom-right: every M0
pull is Cochran-Q incoherent (p = 1e-16..1e-36); M1's are coherent (Q_split p = 0.17).
Two reinterpretations: (1) the "robust kF_sf = 0.93" of earlier pooled fits was absorption of the
excised mismatch, not a measurement; (2) the dat excision is REPRESENTABLE norm physics
misclassified as unknown-unknown because Gate I froze the degenerate norm group — the concrete
motivation for the eigen-combination refinement."""))
        textpage(pdf, "Conclusions, limitations, outlook", W("""
CONCLUSIONS.
1. The mandate's criterion is met: the physical fit extracts understandable physics from foreign
   data (a coherent RES-shape difference; a clean initial-state width) WITHOUT being biased by the
   non-representable component, which it localizes and labels instead of absorbing.
2. Confident pulls from a traditional fit against mismodeled data are provably incoherent
   compromises — the per-bin demand field (Cochran's Q) exposes them with calibrated statistics.
   This held both for the controlled artifact and for real GENIE physics.
3. The excision map is itself the discovery product: "here is where the model cannot describe the
   data, and no coherent parameter variation fixes it" — the operational definition of an
   unknown-unknown candidate.

LIMITATIONS (all logged).
- Gate I freezes whole degenerate groups; representable physics (the GENIE norm offset) can then be
  misclassified as unknown-unknown. v3: fit one Fisher eigen-combination per group.
- Gate null calibration is single-seed so far; a multi-seed study should set the p-thresholds.
- The split-fit is prior-anchored (conservative); diagonal errors are required for exact per-bin
  coherence calibration (correlated systematics need a whitened generalization).
- ADoNIS bank MC (1.87M events) is folded into the error model; at finer binnings it becomes the
  floor (measured: 5.9%/bin at 775 bins vs 1-2.6% at 20 bins).

PROVENANCE. All results reproducible from the branch: scripts/altgen/physical_fit.py (Gate I),
physical_fit_run.py (M0/M1/M2 + modes), run outputs output/altgen/physfit_*.npz, evidence chain in
docs/logbook/altgen_fakedata.md §15-17. GENIE sample: LUCiD container, 3M CCQERES events, seed
20250707. Fits use the frozen 1.87M ADoNIS bank with exact differentiable reweights (no Taylor)."""))
    log(f"[pdf] {pdf_path}")
    log("done")


if __name__ == "__main__":
    main()
