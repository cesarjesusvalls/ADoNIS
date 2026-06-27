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
    plab, chan = [], []
    for f in glob.glob(f"output/achilles/_beam_{name}_vtx*.log") + (
            glob.glob("output/achilles/_pi0beam_vtx*.log") if name == "pi0" else []):
        for line in open(f):
            if not line.startswith("VTX "):
                continue
            m = dict(re.findall(r"(\w+)=(-?[\d.eE+]+)", line))
            ip = int(m.get("inc_pid", 0))
            want = {"pip": 211, "pi0": 111, "pim": -211, "p": 2212, "n": 2112}[name]
            if ip != want:
                continue
            plab.append(float(m["inc_p"])); chan.append(int(m["channel"]))
    return np.array(plab), np.array(chan)


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


def main():
    fig, axes = plt.subplots(5, 2, figsize=(13, 17))
    pion_curves = None
    for r, (pid, name, lbl, kind) in enumerate(ROW):
        a0, a1 = axes[r, 0], axes[r, 1]
        is_pion = pid in _CH
        plo, phi = (80, 900) if is_pion else (200, 1700)
        ped = np.linspace(plo, phi, 11)
        # ADoNIS
        af = Path(f"/tmp/chan_ado_{pid}.npz")
        if af.exists():
            d = np.load(af); ap, aw, ak = d["plab"], d["W"], d["kind"]
            asc = (ak == 0) | (ak == 1); ac2 = ak == 1
            xc, fa, ea = binfrac(ap, asc, ac2, ped)
            a0.errorbar(xc, fa, ea, fmt='s-', color='C0', capsize=3, label='ADoNIS')
            Wed = np.arange(*( (1120, 1620, 40) if is_pion else (1900, 2500, 60)))
            wc, fw, ew = binfrac(aw, asc, ac2, Wed)
            a1.errorbar(wc, fw, ew, fmt='s', color='C0', capsize=3, label='ADoNIS (post-Pauli)')
        # ADoNIS no-Pauli (pi0 only) -> should match pre-Pauli curve
        npf = Path(f"/tmp/chan_ado_{pid}_nopauli.npz")
        if npf.exists():
            d = np.load(npf); np_w, np_k = d["W"], d["kind"]; nsc = (np_k == 0) | (np_k == 1); nc2 = np_k == 1
            Wed = np.arange(1120, 1620, 40); wc, fw2, ew2 = binfrac(np_w, nsc, nc2, Wed)
            a1.errorbar(wc, fw2, ew2, fmt='^', color='C2', capsize=3, label='ADoNIS (no Pauli)')
        # ACHILLES
        plab, chan = ach_load(name)
        if len(chan):
            sc = np.isin(chan, [1, 2]); c2 = chan == 2
            xc, fc, ec = binfrac(plab, sc, c2, ped)
            a0.errorbar(xc, fc, ec, fmt='D--', color='0.35', capsize=3, label='ACHILLES')
        # pre-Pauli cross-section curve
        if is_pion:
            if pion_curves is None:
                Wg = np.linspace(1120, 1600, 80); pion_curves = (Wg, pion_cex_curve(Wg))
            Wg, pc = pion_curves
            a1.plot(Wg, pc[_CH[pid]], '-', color='C3', lw=2, label='cross section (pre-Pauli)')
        else:
            Wg = np.linspace(1950, 2480, 80)
            a1.plot(Wg, nucleon_inel_curve(Wg), '-', color='C3', lw=2, label='cross section (pre-Pauli)')
        ylab = "CEX / (el+cex)" if kind == "cex" else "inel / (el+inel)"
        a0.set_title(f"{lbl} on $^{{12}}$C : {ylab} vs lab |p|"); a0.set_xlabel("lab |p| [MeV]")
        a0.set_ylabel(ylab); a0.legend(fontsize=8)
        a1.set_title(f"{lbl} : {ylab} vs W"); a1.set_xlabel("W [MeV]"); a1.legend(fontsize=8)
    fig.suptitle("Cascade per-W channel fractions: ADoNIS vs ACHILLES (pi: CEX; N: inelastic)", fontsize=13)
    fig.tight_layout(); out = "output/figures/cascade_channel_perW.png"; fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
