"""Anatomy of the QE post-FSI NEUTRON-multiplicity ACH/ADO residual (~+2% in mean N(n)), in the T2K ratio
style (main dsigma + ACH/ADO ratio + chi2/ndf).  One panel per mechanism, all from the SAME two banks
(ADoNIS output/adonis/t2k_C_cc0pi.npz ; ACHILLES output/achilles/t2k_cc1pi_rich_ach_FSI_proc.npz, proc==200),
QE / carbon, identical muon acceptance.  No new generation.

The 4 mechanisms (why mean N(n) disagrees ~2% while per-vertex channel fractions and sigma agree <1%):
  (1) knockout AVALANCHE control : N(p) multiplicity (primary proton + secondary knockout) -> matches,
      so the residual is neutron-specific, not a global avalanche-rate miscount.
  (2) low-|p| / Pauli / recapture : leading-neutron |p| spectrum near threshold (recaptured nucleons are
      set to REST in ADoNIS and dropped from N(n); a threshold-region ratio exposes Pauli/recapture).
  (3) multi-generation HIGH-k tail : N(n) multiplicity distribution -- the deficit grows with k
      (step-resolution / sub-cascade sequencing under-converged at coarse dx).
  (4) NN charge exchange (np->pn)  : in QE the primary nucleon is a PROTON, so EVERY final neutron is
      FSI-generated; P(N(n)>=1) vs leading-proton |p| is the charge-exchange+knockout efficiency vs momentum.

Run: python -u scripts/neutron_multiplicity_anatomy.py [out.png]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.workflow.plotting import chi2_ratio_panel

OUT = sys.argv[1] if len(sys.argv) > 1 else "output/figures/neutron_multiplicity_anatomy.png"
ADO = "output/adonis/t2k_C_cc0pi.npz"
ACH = "output/achilles/t2k_cc1pi_rich_ach_FSI_proc.npz"
COS70 = float(np.cos(np.deg2rad(70.0)))


def muacc(m):
    p = np.linalg.norm(m[:, 1:], axis=1); cz = m[:, 3] / np.clip(p, 1e-9, None)
    return (p > 250) & (p < 7000) & (cz > COS70)


def load():
    a = np.load(ADO, allow_pickle=True)
    h = np.load(ACH, allow_pickle=True)
    ma = muacc(a["mu"]) & (a["w"] > 0)
    hs = (h["proc"] == 200)
    mh = muacc(h["mu"][hs]) & (h["w"][hs] > 0)
    A = dict(w=a["w"][ma], nn=a["n_n"][ma], npr=a["n_p"][ma],
             neut=a["neut"][ma], prot=a["prot"][ma])
    H = dict(w=(h["w"][hs] * float(h["weight_to_nb"]))[mh],
             neut=h["neut_p4"][hs][mh], prot=h["prot_p4"][hs][mh])
    H["nn"] = (np.linalg.norm(H["neut"][:, :, 1:], axis=2) > 1e-6).sum(1)
    H["npr"] = (np.linalg.norm(H["prot"][:, :, 1:], axis=2) > 1e-6).sum(1)
    return A, H


def lead_p(p4):  # |p| of the highest-|p| nucleon per event (0 if none)
    mag = np.linalg.norm(p4[:, :, 1:], axis=2)
    return mag.max(1)


def rank0_p(p4):  # |p| of leading (rank-0) nucleon, NaN-dropped by >0 mask downstream
    return np.linalg.norm(p4[:, 0, 1:], axis=1)


def frac_panel(am, ar, edges, ado, ach, xlabel, title):
    """P(condition) vs binned x, ACHILLES & ADoNIS, with ACH/ADO ratio + chi2/ndf.  ado/ach: (x, sel, w)."""
    def binit(x, sel, w):
        c, f, e = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi); sw = w[m].sum()
            if sw <= 0:
                c.append(0.5 * (lo + hi)); f.append(np.nan); e.append(0); continue
            p = w[m & sel].sum() / sw
            neff = sw ** 2 / max((w[m] ** 2).sum(), 1e-30)
            c.append(0.5 * (lo + hi)); f.append(p); e.append(np.sqrt(max(p * (1 - p), 0) / max(neff, 1)))
        return np.array(c), np.array(f), np.array(e)
    cd, fd, ed = binit(*ado); cc, fc, ec = binit(*ach)
    am.step(np.append(edges[0], edges[1:]), np.append(fc[0], fc), where="pre", color="0.3", lw=1.3, label="ACHILLES")
    am.errorbar(cd, fd, ed, fmt="s", color="C0", ms=3, capsize=2, lw=0.9, label="ADoNIS")
    am.set_title(title, fontsize=9); am.set_ylim(bottom=0); am.legend(fontsize=7)
    am.set_xlim(edges[0], edges[-1])
    m = np.isfinite(fd) & np.isfinite(fc) & (fd > 0) & (fc > 0)
    ar.axhspan(0.9, 1.1, color="green", alpha=0.12); ar.axhline(1.0, ls="--", color="green", lw=0.7)
    if m.any():
        r = fc[m] / fd[m]; er = r * np.sqrt((ed[m] / fd[m]) ** 2 + (ec[m] / fc[m]) ** 2)
        ar.errorbar(cc[m], r, er, fmt="o", color="C3", ms=3, capsize=2, lw=0.8)
        chi2 = float(np.sum((fc[m] - fd[m]) ** 2 / (ed[m] ** 2 + ec[m] ** 2))); ndf = int(m.sum())
        ar.text(0.04, 0.83, f"{chi2/max(ndf,1):.1f}", transform=ar.transAxes, fontsize=9)
    ar.set_ylim(0.5, 1.6); ar.set_xlabel(xlabel, fontsize=8); ar.set_xlim(edges[0], edges[-1])


def main():
    A, H = load()
    print("ADO sel N=%d sigma=%.4e | ACH sel N=%d sigma=%.4e | ACH/ADO sigma=%.4f"
          % (len(A["w"]), A["w"].sum(), len(H["w"]), H["w"].sum(), H["w"].sum() / A["w"].sum()), flush=True)
    print("mean N(n) ADO=%.4f ACH=%.4f ACH/ADO=%.4f | mean N(p) ADO=%.4f ACH=%.4f ACH/ADO=%.4f"
          % (np.average(A["nn"], weights=A["w"]), np.average(H["nn"], weights=H["w"]),
             np.average(H["nn"], weights=H["w"]) / np.average(A["nn"], weights=A["w"]),
             np.average(A["npr"], weights=A["w"]), np.average(H["npr"], weights=H["w"]),
             np.average(H["npr"], weights=H["w"]) / np.average(A["npr"], weights=A["w"])), flush=True)

    fig, ax = plt.subplots(2, 4, figsize=(20, 6.6), height_ratios=[3, 1], sharex="col")

    # (3) N(n) multiplicity -- the headline high-k tail
    medg = np.arange(-0.5, 5.5, 1.0)
    chi2_ratio_panel(ax[0, 0], ax[1, 0], medg, {"values": H["nn"], "w": H["w"]},
                     {"values": A["nn"], "w": A["w"]}, label="N(n) per event  [(3) high-k tail]",
                     ado_label="ADoNIS", ref_label="ACHILLES", logy=True, ratio_ylim=(0.5, 2.0))
    ax[1, 0].set_xlabel("N(neutrons) [(3) multi-generation tail]", fontsize=8)

    # (1) N(p) multiplicity -- avalanche control (should match)
    chi2_ratio_panel(ax[0, 1], ax[1, 1], medg, {"values": H["npr"], "w": H["w"]},
                     {"values": A["npr"], "w": A["w"]}, label="N(p) per event  [(1) avalanche control]",
                     ado_label="ADoNIS", ref_label="ACHILLES", logy=True, ratio_ylim=(0.5, 2.0))
    ax[1, 1].set_xlabel("N(protons) [(1) knockout avalanche]", fontsize=8)

    # (2) leading-neutron |p| spectrum -- low-|p| Pauli / recapture region
    pedg = np.linspace(0, 800, 33)
    pna = rank0_p(A["neut"]); pnh = rank0_p(H["neut"])
    ka = pna > 0; kh = pnh > 0
    chi2_ratio_panel(ax[0, 2], ax[1, 2], pedg, {"values": pnh[kh], "w": H["w"][kh]},
                     {"values": pna[ka], "w": A["w"][ka]}, label="leading-neutron |p|  [(2) Pauli/recap]",
                     ado_label="ADoNIS", ref_label="ACHILLES", logy=True, ratio_ylim=(0.5, 1.6))
    ax[1, 2].set_xlabel("leading-neutron |p| [MeV]  [(2) low-|p| Pauli]", fontsize=8)
    ax[0, 2].axvline(200, color="purple", ls=":", lw=1.0); ax[1, 2].axvline(200, color="purple", ls=":", lw=1.0)

    # (4) P(N(n)>=1) vs leading-proton |p| -- charge-exchange / knockout efficiency (QE: all n are FSI)
    qedg = np.linspace(0, 1200, 13)
    lpa = lead_p(A["prot"]); lph = lead_p(H["prot"])
    frac_panel(ax[0, 3], ax[1, 3], qedg,
               (lpa[lpa > 0], (A["nn"] >= 1)[lpa > 0], A["w"][lpa > 0]),
               (lph[lph > 0], (H["nn"] >= 1)[lph > 0], H["w"][lph > 0]),
               "leading-proton |p| [MeV]  [(4) np->pn]", "P(N(n)$\\geq$1) vs lead-p |p|  [(4) charge exch.]")
    ax[0, 3].set_ylabel("P(N(n)$\\geq$1)", fontsize=8)
    ax[0, 0].set_ylabel("d$\\sigma$ [nb]", fontsize=8); ax[1, 0].set_ylabel("ACH/ADO", fontsize=8)
    ax[0, 2].set_ylabel("d$\\sigma$/d|p| [nb/MeV]", fontsize=8)
    fig.suptitle("QE CC0$\\pi$ on $^{12}$C: anatomy of the post-FSI neutron-multiplicity ACH/ADO residual "
                 "(σ matches 0.4%; N(p) matches; N(n) tail-deficit is neutron-specific)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(OUT, dpi=120)
    print("wrote", OUT, flush=True)

    # numeric tail table
    def fr(c, w, nm=4):
        c = np.clip(c, 0, nm); return np.array([w[c == k].sum() for k in range(nm + 1)]) / w.sum()
    fa, fh = fr(A["nn"], A["w"]), fr(H["nn"], H["w"])
    print("N(n)=k   ADO       ACH       ACH/ADO", flush=True)
    for k in range(5):
        print("  %d   %.5f  %.5f   %.3f" % (k, fa[k], fh[k], fh[k] / max(fa[k], 1e-30)), flush=True)


if __name__ == "__main__":
    main()
