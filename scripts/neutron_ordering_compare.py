"""Forward QE CC0pi N(n)/N(p) multiplicity: ADoNIS PRE-change (RNG-order) vs POST-change (ACHILLES
creation-order, M=1) vs ACHILLES -- the headline test of whether matching ACHILLES's within-step
processing order closes the neutron-multiplicity residual.  Muon-accepted, carbon.  Fractions of sigma
(scale-invariant: the _ord bank is 6 batches, the old bank is one file; both normalized to their own total).

Run: python -u scripts/neutron_ordering_compare.py [out.png]
"""
import sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "output/figures/neutron_ordering_compare.png"
OLD = "output/adonis/t2k_C_cc0pi.npz"
NEW_GLOB = "output/adonis/t2k_C_cc0pi_cx_batch*.npz"   # M=1 ACHILLES-order + NN charge exchange
ACH = "output/achilles/t2k_cc1pi_rich_ach_FSI_proc.npz"
COS70 = float(np.cos(np.deg2rad(70.0)))


def muacc(m):
    p = np.linalg.norm(m[:, 1:], axis=1); cz = m[:, 3] / np.clip(p, 1e-9, None)
    return (p > 250) & (p < 7000) & (cz > COS70)


def load_old():
    d = np.load(OLD, allow_pickle=True); m = muacc(d["mu"]) & (d["w"] > 0)
    return d["n_n"][m], d["n_p"][m], d["w"][m]


def load_new():
    NN, NP, W = [], [], []
    for f in sorted(glob.glob(NEW_GLOB)):
        d = np.load(f, allow_pickle=True); m = muacc(d["mu"]) & (d["w"] > 0)
        NN.append(d["n_n"][m]); NP.append(d["n_p"][m]); W.append(d["w"][m])
    return np.concatenate(NN), np.concatenate(NP), np.concatenate(W)


def load_ach():
    h = np.load(ACH, allow_pickle=True); hs = h["proc"] == 200
    m = muacc(h["mu"][hs]) & (h["w"][hs] > 0); w = (h["w"][hs] * float(h["weight_to_nb"]))[m]
    nn = (np.linalg.norm(h["neut_p4"][hs][m][:, :, 1:], axis=2) > 1e-6).sum(1)
    npr = (np.linalg.norm(h["prot_p4"][hs][m][:, :, 1:], axis=2) > 1e-6).sum(1)
    return nn, npr, w


def fr(c, w, nm=5):
    c = np.clip(c, 0, nm)
    f = np.array([w[c == k].sum() for k in range(nm + 1)]) / w.sum()
    neff = np.array([(w[c == k].sum() ** 2) / max((w[c == k] ** 2).sum(), 1e-30) for k in range(nm + 1)])
    e = np.sqrt(np.clip(f * (1 - f), 0, None) / np.clip(neff, 1, None))
    return f, e


def panel(am, ar, lab, ado_old, ado_new, ach, w_old, w_new, w_ach, nm=5):
    x = np.arange(nm + 1)
    fo, eo = fr(ado_old, w_old, nm); fn, en = fr(ado_new, w_new, nm); fh, eh = fr(ach, w_ach, nm)
    am.errorbar(x, fh, eh, fmt='D--', color='0.3', ms=5, capsize=2, lw=1.1, label='ACHILLES')
    am.errorbar(x - 0.06, fo, eo, fmt='s', color='C1', ms=4, capsize=2, lw=0.9, label='ADoNIS (RNG order)')
    am.errorbar(x + 0.06, fn, en, fmt='o', color='C0', ms=4, capsize=2, lw=0.9, label='ADoNIS (M=1 + NN charge-exch)')
    am.set_yscale('log'); am.set_ylim(max(1e-5, fh[fh > 0].min() * 0.3), 1.4)
    am.set_ylabel("fraction of $\\sigma$", fontsize=9); am.set_title(lab, fontsize=10); am.legend(fontsize=8)
    ar.axhspan(0.9, 1.1, color='green', alpha=0.12); ar.axhline(1.0, ls='--', color='green', lw=0.7)
    mo = (fo > 0) & (fh > 0); mn = (fn > 0) & (fh > 0)
    ar.errorbar(x[mo] - 0.06, fh[mo] / fo[mo], fmt='s', color='C1', ms=4, capsize=2,
                yerr=(fh[mo] / fo[mo]) * np.sqrt((eo[mo] / fo[mo]) ** 2 + (eh[mo] / fh[mo]) ** 2))
    ar.errorbar(x[mn] + 0.06, fh[mn] / fn[mn], fmt='o', color='C0', ms=4, capsize=2,
                yerr=(fh[mn] / fn[mn]) * np.sqrt((en[mn] / fn[mn]) ** 2 + (eh[mn] / fh[mn]) ** 2))
    ar.set_ylim(0.5, 2.0); ar.set_ylabel("ACH/ADO", fontsize=8); ar.set_xlabel(lab, fontsize=9)
    # means
    mo_ = np.average(ado_old, weights=w_old); mn_ = np.average(ado_new, weights=w_new); mh_ = np.average(ach, weights=w_ach)
    am.text(0.02, 0.04, f"mean: ACH {mh_:.4f}\nRNG {mh_/mo_:.3f}  ACH-ord {mh_/mn_:.3f}",
            transform=am.transAxes, fontsize=8, va='bottom')


def main():
    no, po, wo = load_old(); nn, pn, wn = load_new(); nh, ph, wh = load_ach()
    print(f"N events: old={len(wo)} new={len(wn)} ach={len(wh)}", flush=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), height_ratios=[3, 1])
    panel(ax[0, 0], ax[1, 0], "N(neutrons)", no, nn, nh, wo, wn, wh)
    panel(ax[0, 1], ax[1, 1], "N(protons)", po, pn, ph, wo, wn, wh)
    fig.suptitle("QE CC0$\\pi$/$^{12}$C post-FSI multiplicity: ADoNIS old vs M=1+NN-charge-exchange vs ACHILLES",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(OUT, dpi=120); print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
