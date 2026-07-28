"""Validation figure for the tagged-beam samples: ADoNIS vs ACHILLES sigma(p), ratio panel + chi2/ndf.

One column per beam (pi+ / p / n on 12C), two rows of observables:
    row 1: sigma_reaction(p)
    row 2: sigma_absorption(p)     [pi+]      |  sigma_pion-production(p)  [p, n]
Both sides use the SAME in-medium cross sections and the SAME CrossSection-mode beam geometry, so this is
a pure TRANSPORT comparison -- the cascade's own validation, with no hard vertex and no spectral function
anywhere in it.

Usage:  python -m analysis.paper.beams.make_figs [--nbins 15]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper.beams import beam_bank as BB, achilles_beam as AB       # noqa: E402
from analysis.paper import style                                       # noqa: E402

BEAMS = ("pip", "prot", "neut")
BEAM_TEX = {"pip": "$\\pi^+$", "prot": "p", "neut": "n"}
NUCLEUS_TEX = {"C": "$^{12}$C", "Ar": "$^{40}$Ar"}
SECOND = {"pip": "absorption", "prot": "$\\pi$ production", "neut": "$\\pi$ production"}
# Which ADoNIS cascade build each bank suffix corresponds to -- stamped on the figure so two
# otherwise-identical-looking plots can never be confused.
BUILD = {"": "BASELINE build (no pion birth-position fix)",
         "_bpfix": "WITH pion birth-position fix (D7)"}


def adonis_sigma(beam, nbins, target="C", suffix=""):
    # `suffix` selects an alternative bank build (e.g. "_bpfix" = pion birth-position fix).  A beam that
    # has no suffixed bank falls back to the canonical one, so a partial regeneration still plots.
    import os
    pattern = os.environ.get("ADONIS_BEAM_PATTERN", "output/beam_{beam}_{target}{suffix}")
    path = pattern.format(beam=beam, target=target, suffix=suffix)
    if suffix and not Path(path).is_dir():
        path = pattern.format(beam=beam, target=target, suffix="")
    B = BB.load(path)
    man = B["manifest"]
    adonis_sigma.last = (path, int(man.get("n_total", 0)))    # for labelling the panel
    p = np.asarray(B["beam_p"], float)
    edges = np.linspace(man["pmin"], man["pmax"], nbins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, nbins - 1)
    ntry = np.bincount(idx, minlength=nbins).astype(float)
    from adonis.workflow import records as REC
    _fl = REC.derive_flags(B)                            # reacted/absorbed derived from prim_fate + nsc_prim
    react = _fl["reacted"].astype(float)
    second = _fl["absorbed"].astype(float) if man["species"] == "PION" \
        else (np.asarray(B["n_pi_out"]) > 0).astype(float)
    nr = np.bincount(idx, weights=react, minlength=nbins)
    ns = np.bincount(idx, weights=second, minlength=nbins)
    PIR2 = man["pir2_mb"]
    with np.errstate(divide="ignore", invalid="ignore"):
        sr, ss = PIR2 * nr / ntry, PIR2 * ns / ntry
        er, es = PIR2 * np.sqrt(nr) / ntry, PIR2 * np.sqrt(ns) / ntry
    return edges, sr, ss, er, es


def chi2(a, b, ea, eb):
    m = np.isfinite(a) & np.isfinite(b) & ((ea > 0) | (eb > 0)) & (a + b > 0)
    if not m.any():
        return np.nan, 0
    d = (a[m] - b[m]) ** 2 / (ea[m] ** 2 + eb[m] ** 2)
    return float(d.sum()), int(m.sum())


def main(nbins=15, target="C", suffix=""):
    style.use()
    fig, axes = plt.subplots(4, 3, figsize=(13.5, 10.5), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1.2, 3, 1.2], "hspace": 0.08, "wspace": 0.22})
    for c, beam in enumerate(BEAMS):
        try:
            edges, ar, as_, aer, aes = adonis_sigma(beam, nbins, target, suffix)
        except Exception as e:                                   # bank not built yet
            for r in range(4):
                axes[r, c].text(.5, .5, f"{beam}: {e}", ha="center", va="center", fontsize=7,
                                transform=axes[r, c].transAxes)
            continue
        hr, hs, her, hes, _nr, _ns, ntried = AB.sigma_of_p(beam, edges, target)
        cen = 0.5 * (edges[:-1] + edges[1:])
        # SAME visual conventions as the neutrino panels (adonis/workflow/plotting.chi2_ratio_panel):
        # ACHILLES = grey step + stat band, ADoNIS = blue squares, ratio ACH/ADO = red dots on a green band.
        for row, (A, EA, H, EH, lab) in enumerate(((ar, aer, hr, her, "reaction"),
                                                   (as_, aes, hs, hes, SECOND[beam]))):
            ax, rx = axes[2 * row, c], axes[2 * row + 1, c]
            ax.fill_between(edges, np.append(H - EH, (H - EH)[-1]), np.append(H + EH, (H + EH)[-1]),
                            step="post", color="0.55", alpha=0.55, lw=0, label="ACH stat")
            ax.step(edges, np.append(H, H[-1]), where="post", color="0.3", lw=1.3, label="ACHILLES")
            ax.errorbar(cen, A, yerr=EA, fmt="s", color="C0", ms=3, capsize=2, lw=0.9, label="ADoNIS")
            x2, nd = chi2(A, H, EA, EH)
            ax.set_ylabel(f"$\\sigma$ [mb] — {lab}")
            if row == 0:
                _bp, _bn = getattr(adonis_sigma, "last", ("?", 0))
                _tag = "post-fix" if _bp.endswith("_bpfix") else "baseline"
                ax.set_title(f"{BEAM_TEX[beam]} + {NUCLEUS_TEX[target]}   "
                             f"(ACHILLES {int(ntried):,} tried | ADoNIS {_bn/1e6:.2f}M ev, {_tag})",
                             fontsize=8)
            ax.set_ylim(bottom=0)
            ax.text(.03, .90, f"$\\chi^2$/ndf {x2/max(nd,1):.2f}", transform=ax.transAxes, fontsize=8)
            if row == 0 and c == 0:
                ax.legend(fontsize=7)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = H / A                                                     # ACH/ADO, as in the nu panels
                er = np.abs(r) * np.sqrt((EA / np.where(A > 0, A, np.nan)) ** 2
                                         + (EH / np.where(H > 0, H, np.nan)) ** 2)
            m = np.isfinite(r) & (A > 0) & (H > 0)
            rx.axhspan(0.95, 1.05, color="green", alpha=0.12)
            rx.axhline(1.0, ls="--", color="green", lw=0.7)
            rx.errorbar(cen[m], r[m], yerr=er[m], fmt="o", color="C3", ms=3, capsize=2, lw=0.8)
            rx.set_ylim(0.8, 1.2)
            rx.text(0.04, 0.78, f"{x2/max(nd,1):.1f}", transform=rx.transAxes, fontsize=9)
            if row == 1:
                rx.set_xlabel("tagged beam $|p|$ [MeV/c]")
    fig.suptitle(f"Tagged-beam validation: ADoNIS vs ACHILLES on {NUCLEUS_TEX[target]}  —  pure TRANSPORT "
                 "(no hard vertex, no spectral function).  Band = $\\pm$5%\n"
                 f"ADoNIS cascade: {BUILD.get(suffix, suffix)}"
                 + ("   [only the p column differs between builds; $\\pi^+$/n reuse the baseline banks]"
                    if suffix else ""),
                 fontsize=10)
    style.save(fig, f"beams_validation_{target}{suffix}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbins", type=int, default=15)
    ap.add_argument("--target", default="both", help="C | Ar | both")
    ap.add_argument("--suffix", default="", help="bank suffix, e.g. _bpfix")
    a = ap.parse_args()
    targets = ["C", "Ar"] if a.target == "both" else [a.target]
    for _t in targets:
        main(nbins=a.nbins, target=_t, suffix=a.suffix)
