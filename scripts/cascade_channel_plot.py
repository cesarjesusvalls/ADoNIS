"""Combined per-W channel plot (pions: CEX fraction; nucleons: inelastic fraction), ADoNIS vs ACHILLES.
Reads /tmp/chan_ado_<pid>.npz (+ _nopauli) and output/achilles/_beam_<name>_vtx*.log.
Col 1: chan2 fraction vs lab |p| (ADoNIS vs ACHILLES).  Col 2: vs W (runtime post-Pauli + pre-Pauli
cross-section curve; for pi0 also the no-Pauli runtime, to show the differential-Pauli effect)."""
import re, glob, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adonis.fsi.mb import cascade_mb as mb
from adonis.fsi.nucleon_cascade import nn_elastic_sigma
from adonis.fsi import nn_inelastic as nni
from adonis.constants import mp as _MP, mn as _MN
import jax.numpy as jnp


def nucleon_inel_curve(W):
    """nucleon-summed inelastic fraction sigma_inel/(sigma_el+sigma_inel) vs W [MeV] (p-beam == n-beam)."""
    W = np.asarray(W, float); s = (W ** 2)
    num = np.zeros_like(W); den = np.zeros_like(W)
    for same in (True, False):                                   # target = same-iso (pp/nn) or opposite (pn)
        mpair = (0.5 * (_MP + _MN) / 1000.0) if not same else (_MP / 1000.0)
        se = np.asarray(nn_elastic_sigma(jnp.asarray(W), jnp.full(len(W), same), jnp.full(len(W), mpair)))
        pcm = np.sqrt(np.clip(s / 4.0 - ((_MP + _MN) / 2) ** 2, 1e-6, None)) / 1000.0
        si = np.asarray(nni.sigma_nn_ndelta(jnp.asarray(W) / 1000.0, jnp.asarray(pcm), jnp.full(len(W), same)))
        se = np.clip(se, 0, None); si = np.clip(si, 0, None)
        num += si; den += se + si
    return num / np.clip(den, 1e-9, None)

ROW = [(211, "pip", r"$\pi^+$", "cex"), (111, "pi0", r"$\pi^0$", "cex"), (-211, "pim", r"$\pi^-$", "cex"),
       (2212, "p", "p", "inel"), (2112, "n", "n", "inel")]
_CH = {211: 0, 111: 1, -211: 2}


def ach_load(name):
    plab, W, chan = [], [], []
    want = {"pip": 211, "pi0": 111, "pim": -211, "p": 2212, "n": 2112}[name]
    for f in glob.glob(f"output/achilles/_wbeam_{name}_vtx*.log"):   # W-instrumented beams (achilles:vertexw)
        for line in open(f):
            if not line.startswith("VTX "):
                continue
            m = dict(re.findall(r"(\w+)=(-?[\d.]+(?:[eE][-+]?\d+)?)", line))
            if int(m.get("inc_pid", 0)) != want:
                continue
            plab.append(float(m["inc_p"])); W.append(float(m.get("W", -1))); chan.append(int(m["channel"]))
    return np.array(plab), np.array(W), np.array(chan)


def binfrac(plb, sc, c2, edges):
    f, e, c = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = sc & (plb >= lo) & (plb < hi); n = int(m.sum())
        if n < 40:
            f.append(np.nan); e.append(0); c.append(0.5 * (lo + hi)); continue
        fr = c2[m].sum() / n; f.append(fr); e.append(np.sqrt(fr * (1 - fr) / n)); c.append(0.5 * (lo + hi))
    return np.array(c), np.array(f), np.array(e)


def pion_cex_curve(W):  # nucleon-summed sigma_cex/sigma_scat for a pion charge index
    out = {}
    for ci in (0, 1, 2):
        sp = np.asarray(mb.jax_channel_sigmas_resolved(jnp.asarray(W), jnp.full(len(W), ci), jnp.zeros(len(W), int)))
        sn = np.asarray(mb.jax_channel_sigmas_resolved(jnp.asarray(W), jnp.full(len(W), ci), jnp.ones(len(W), int)))
        cex = (sp.sum(1) - sp[:, ci]) + (sn.sum(1) - sn[:, ci])     # out!=in summed over p,n
        scat = sp.sum(1) + sn.sum(1)
        out[ci] = cex / np.clip(scat, 1e-9, None)
    return out


def panel(am, ar, edges, ado, ach, xlabel, ylab, title):
    """T2K-style: main axis am (ADoNIS + ACHILLES) + ratio axis ar (ADoNIS/ACHILLES) + chi2/ndf."""
    ca, fa, ea = ado; cc, fc, ec = ach
    am.errorbar(cc, fc, ec, fmt='D--', color='0.35', ms=4, capsize=2, lw=1.0, label='ACHILLES')
    am.errorbar(ca, fa, ea, fmt='s-', color='C0', ms=4, capsize=2, lw=1.0, label='ADoNIS')
    am.set_title(title, fontsize=9); am.set_ylabel(ylab, fontsize=8); am.legend(fontsize=7); am.set_ylim(bottom=0)
    m = np.isfinite(fa) & np.isfinite(fc) & (fc > 0) & (fa > 0)
    ar.axhspan(0.9, 1.1, color='green', alpha=0.12); ar.axhline(1.0, ls='--', color='0.5', lw=0.7)
    if m.any():
        r = fa[m] / fc[m]; er = r * np.sqrt((ea[m] / fa[m]) ** 2 + (ec[m] / fc[m]) ** 2)
        ar.errorbar(ca[m], r, er, fmt='o', color='C3', ms=3, capsize=2)
        chi2 = float(np.sum((fa[m] - fc[m]) ** 2 / (ea[m] ** 2 + ec[m] ** 2))); ndf = int(m.sum())
        ar.text(0.03, 0.82, f"$\\chi^2$/ndf = {chi2:.0f}/{ndf} = {chi2/max(ndf,1):.2f}",
                transform=ar.transAxes, fontsize=8)
    ar.set_ylim(0.7, 1.3); ar.set_xlabel(xlabel, fontsize=8); ar.set_ylabel("ADO/ACH", fontsize=7)
    am.set_xlim(edges[0], edges[-1]); ar.set_xlim(edges[0], edges[-1])   # ratio shares the main x-range


def main():
    fig, axes = plt.subplots(10, 2, figsize=(13, 22), height_ratios=[3, 1] * 5)
    for r, (pid, name, lbl, kind) in enumerate(ROW):
        am0, ar0 = axes[2 * r, 0], axes[2 * r + 1, 0]      # left col: vs lab|p| (main, ratio)
        am1, ar1 = axes[2 * r, 1], axes[2 * r + 1, 1]      # right col: vs W (main, ratio)
        is_pion = pid in _CH
        plo, phi = (80, 900) if is_pion else (200, 1700)
        ped = np.linspace(plo, phi, 13)
        Wed = np.arange(*((1120, 1620, 30) if is_pion else (1950, 2480, 40)))
        ylab = "CEX/(el+cex)" if kind == "cex" else "inel/(el+inel)"
        # ADoNIS (charge-matched all-scatter)
        d = np.load(f"/tmp/chan_ado_{pid}.npz"); ap, aw, ak = d["plab"], d["W"], d["kind"]
        if is_pion and "chpre" in d:
            keep = d["chpre"] == _CH[pid]; ap, aw, ak = ap[keep], aw[keep], ak[keep]
        asc = (ak == 0) | (ak == 1); ac2 = ak == 1
        # ACHILLES
        plab, Wa, chan = ach_load(name); sc = np.isin(chan, [1, 2]); c2 = chan == 2; wm = sc & (Wa > 0)
        panel(am0, ar0, ped, binfrac(ap, asc, ac2, ped), binfrac(plab, sc, c2, ped),
              "lab |p| [MeV]", ylab, f"{lbl} on $^{{12}}$C : {ylab} vs lab |p|")
        panel(am1, ar1, Wed, binfrac(aw, asc, ac2, Wed), binfrac(Wa, wm, c2, Wed),
              "W [MeV]", ylab, f"{lbl} : {ylab} vs W (invariant mass)")
    fig.suptitle("Cascade channel fractions, ADoNIS vs ACHILLES on $^{12}$C (pi: charge-exchange; N: inelastic)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99]); out = "output/figures/cascade_channel_perW.png"; fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
