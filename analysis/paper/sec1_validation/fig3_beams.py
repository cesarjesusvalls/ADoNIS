"""Paper Fig 3: pi+ nucleus ABSORPTION + REACTION sigma(p), ADoNIS vs ACHILLES, on 12C and 40Ar.

A sigma-vs-scan (efficiency) figure, NOT a per-event histogram: sigma_X(p_bin) = piR^2 * N_X(bin)/N_tried(bin)
with binomial errors, computed on both sides and drawn through the shared chi2_ratio_panel in CURVE mode.
Reuses the validated beam sigma(p) computation (beams.make_figs.adonis_sigma on the paper_banks_p4 beam
banks + beams.achilles_beam.sigma_of_p on the ACHILLES CrossSection-mode oracle).

Paper layout: rows = nucleus (C top, Ar bottom), cols = absorption (left) / reaction (right).

  python -m analysis.paper.sec1_validation.fig3_beams [--nbins 15]
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ADONIS_BEAM_PATTERN", "output/paper_banks_p4/beam_{beam}_{target}/merged")
from analysis.paper.beams.make_figs import adonis_sigma        # noqa: E402
from analysis.paper.beams import achilles_beam as AB           # noqa: E402
from adonis.workflow.plotting import chi2_ratio_panel          # noqa: E402
from analysis.paper import style                               # noqa: E402
from analysis.paper import plotcache                           # noqa: E402

NUC_TEX = {"C": r"$^{12}$C", "Ar": r"$^{40}$Ar"}
BEAM = "pip"
BANK = "output/paper_banks_p4/beam_{beam}_{target}/merged"


def reduce_nucleus(nuc, nbins):
    """(edges + ADoNIS/ACHILLES sigma & errors) for one nucleus, memoised: this is the 65 s of the
    figure's 74 s -- multi-GB banks and hepmc TEXT scans collapsing to ~30 numbers per curve."""
    def _build():
        edges, sr, ss, er, es = adonis_sigma(BEAM, nbins, nuc)
        hr, hs, her, hes, _nr, _ns, _nt = AB.sigma_of_p(BEAM, edges, nuc)
        return dict(edges=edges, sr=sr, ss=ss, er=er, es=es, hr=hr, hs=hs, her=her, hes=hes)
    d = plotcache.cached(
        f"fig03_{BEAM}_{nuc}_n{nbins}", _build,
        deps=[ROOT / BANK.format(beam=BEAM, target=nuc), *AB.source_paths(BEAM, nuc)],
        params={"beam": BEAM, "target": nuc, "nbins": nbins})
    return (d["edges"], d["sr"], d["ss"], d["er"], d["es"],
            d["hr"], d["hs"], d["her"], d["hes"])


def main(nbins=30, out="fig03_pi_nucleus_sigma"):
    style.use()
    # One block per nucleus (sigma + ratio glued), a real gap BETWEEN blocks: with a single 4-row
    # hspace=0 grid the Ar titles landed on top of the C ratio panels.
    fig = plt.figure(figsize=(7.1, 4.6))
    outer = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28)
    ax = {}
    for ni in range(2):
        for oi in range(2):
            inner = outer[ni, oi].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
            a0 = fig.add_subplot(inner[0])
            ax[ni, oi] = (a0, fig.add_subplot(inner[1], sharex=a0))
            a0.tick_params(labelbottom=False)
    for ni, nuc in enumerate(("C", "Ar")):
        try:
            edges, sr, ss, er, es, hr, hs, her, hes = reduce_nucleus(nuc, nbins)
        except Exception as e:
            for oi in range(2):
                ax[ni, oi][0].text(0.5, 0.5, f"{nuc}: {e}", ha="center", fontsize=7,
                                   transform=ax[ni, oi][0].transAxes)
            continue
        cen = 0.5 * (edges[:-1] + edges[1:]) / 1000.0        # GeV/c (paper axis)
        PMAX = {"absorption": 0.5, "reaction": 1.0}          # paper Fig 3 shown ranges [GeV/c]
        for oi, (A, EA, H, EH, lab) in enumerate(((ss, es, hs, hes, "absorption"),
                                                  (sr, er, hr, her, "reaction"))):
            a0, a1 = ax[ni, oi]
            k = cen <= PMAX[lab]                             # crop to the paper's shown |p| range
            # sigma_X(bin) is a BINNED quantity -> hand the panel the bin edges so it steps like fig01
            # instead of drawing a line through bin centres.  k is a prefix cut, so the edges are too.
            edg = edges[:int(k.sum()) + 1] / 1000.0          # MeV -> GeV/c
            res = chi2_ratio_panel(a0, a1, None,
                                   {"x": cen[k], "y": H[k], "yerr": EH[k], "edges": edg},
                                   {"x": cen[k], "y": A[k], "yerr": EA[k]},
                                   label="",                 # nucleus/channel goes IN the axes
                                   xlabel=r"beam $|p|$ [GeV/c]", ratio_ylim=(0.8, 1.2),
                                   ado_label="ADoNIS", ref_label="ACHILLES",
                                   total_color=style.C_QE,   # single series -> the blue of fig01's QE
                                   ratio_color=style.C_RATIO,
                                   ado_lighten=style.ADO_LIGHTEN, ref_darken=style.REF_DARKEN,
                                   headroom=0.30)
            a0.set_ylabel(r"$\sigma$ [mb]"); a1.set_ylabel("ratio")
            # sigma's y=0 label and the ratio's top label collide across the glued spines -> pin the
            # ratio ticks inside the range.
            a1.set_yticks([0.9, 1.0, 1.1])
            # channel is a COLUMN header (top row only); only the short nucleus tag repeats per panel,
            # top-right, where the falling tail leaves room in both columns.
            if ni == 0:
                a0.set_title(lab, fontsize=9)
                a1.set_xlabel("")                            # x label only on the bottom block
            a0.text(0.97, 0.95, rf"$\pi^+$ {NUC_TEX[nuc]}", transform=a0.transAxes,
                    fontsize=8, va="top", ha="right")
            print(f"  {nuc} {lab:10s} chi2/ndf {res['chi2']/max(res['ndf'],1):6.2f} (ndf={res['ndf']})", flush=True)
    # No legend: the only key would be dashed=ACHILLES / solid=ADoNIS, which the caption states.
    fig.suptitle(r"$\pi^+$ absorption + reaction on $^{12}$C / $^{40}$Ar (pure transport)",
                 fontsize=9, y=0.985)
    style.save(fig, out)


def render(spec=None):
    """Figure-hook entry (analysis/paper/figures). Compute is main()'s, unchanged."""
    p = (spec or {}).get("params", {}) or {}
    main(nbins=int(p.get("nbins", 30)))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbins", type=int, default=30)
    main(nbins=ap.parse_args().nbins)
