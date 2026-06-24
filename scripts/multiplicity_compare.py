"""Per-channel final-state particle-multiplicity: dsigma vs N(p), N(n), N(pi+), N(pi0), N(pi-), QE-only
and RES-only separately, ADoNIS (new standard: P=2, full-batch n_w=0, queue) vs ACHILLES (proc-filtered:
QE=200, RES=401+402).  ADoNIS counts the FULL post-FSI escaped final state from the engine's out buffer
(nterms species+charge); each channel normalized to its OWN sigma (fraction) so shapes compare directly.
ACHILLES proc bank has protons + pions but NO neutrons -> N(n) is ADoNIS-only.

Run: python -u scripts/multiplicity_compare.py [N_per_channel=50000] [out.png]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
np.seterr(all="ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.workflow.materials import resolve_targets
from adonis.xsec import res_xsec, qe_xsec
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec.spectral import SpectralFunction
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _load_density
from adonis.fsi.cascade_full import cascade_nucleus, PION, NUCLEON

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/multiplicity_QE_RES_C.png"
SPECIES = [("N(p)", NUCLEON, 1, 2212), ("N(n)", NUCLEON, 0, None),
           ("N(pi+)", PION, 0, 211), ("N(pi0)", PION, 1, 111), ("N(pi-)", PION, 2, -211)]
ACH_PROC = {"QE": [200], "RES": [401, 402]}


def _cfg():
    tg = resolve_targets("C")[0][0]
    _, _, _, r = _load_density(tg.density_p, tg.density_n)
    ms = max(600, int(np.ceil(3.0 * r / 0.04)))
    return tg, DiscreteCascadeConfig(step=0.04, max_steps=ms, seed=1, nn_inelastic=True,
                                     nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)


def _count(nt):
    sp = np.asarray(nt[0]["species"]); ch = np.asarray(nt[0]["charge"]); al = np.asarray(nt[0]["alive"])
    return {lab: ((sp == s) & (ch == c) & al).sum(1) for lab, s, c, _ in SPECIES}


def ado_res(P=2, n_w=0):
    tg, cfg = _cfg(); sfn, sfp = SpectralFunction(tg.spectral_n), SpectralFunction(tg.spectral_p)
    e = res_xsec.generate(N, seed=0, return_events=True, sf_n=sfn, sf_p=sfp, n_neutron=6, n_proton=6)["events"]
    rs = np.asarray(e["w"]) > 0
    _, nt, _, _ = cascade_nucleus(jnp.asarray(np.asarray(e["p_pi"])[rs]), jnp.asarray(np.asarray(e["p_N"])[rs]),
        jnp.asarray(np.asarray(e["ppid"])[rs], jnp.int32), jnp.asarray(np.asarray(e["ipid"])[rs], jnp.int32),
        jnp.asarray(np.asarray(e["Npid"])[rs], jnp.int32), cfg, jax.random.PRNGKey(11), P=P, channel="res", n_w=n_w)
    return _count(nt), np.asarray(e["w"])[rs]


def ado_qe(P=2, n_w=0):
    tg, cfg = _cfg(); sfn = SpectralFunction(tg.spectral_n)
    q = qe_xsec.sample_importance(N, seed=0, sf=sfn, n_neutron=6); m = np.asarray(q["w"]) != 0; nqe = int(m.sum())
    _, nt, _, _ = cascade_nucleus(jnp.zeros((nqe, 4)), jnp.asarray(np.asarray(q["p_out"])[m]),
        jnp.zeros(nqe, jnp.int32), jnp.full(nqe, 2112, jnp.int32), jnp.full(nqe, 2212, jnp.int32),
        cfg, jax.random.PRNGKey(13), P=P, channel="qe", n_w=n_w)
    return _count(nt), np.asarray(q["w"])[m]


def ach_channel(procs):
    d = np.load("data/oracle/t2k_cc1pi_rich_ach_FSI_proc.npz", allow_pickle=True)
    sel = np.isin(np.asarray(d["proc"]), procs); w = np.asarray(d["w"])[sel]
    pp4 = np.asarray(d["prot_p4"])[sel]; pip = np.asarray(d["pi_pid"])[sel]
    n_p = (np.linalg.norm(pp4[:, :, 1:], axis=2) > 1e-6).sum(1)
    return {"N(p)": n_p, "N(pi+)": (pip == 211).sum(1), "N(pi0)": (pip == 111).sum(1),
            "N(pi-)": (pip == -211).sum(1)}, w


def frac(counts, w, nmax=6):
    c = np.clip(counts, 0, nmax)
    return np.array([w[c == k].sum() for k in range(nmax + 1)]) / w.sum()


def main():
    print("[mult] ADoNIS RES (P=2) ...", flush=True); aR, wR = ado_res()
    print("[mult] ADoNIS QE  (P=2) ...", flush=True); aQ, wQ = ado_qe()
    print("[mult] ACHILLES QE/RES ...", flush=True)
    hQ, hwQ = ach_channel(ACH_PROC["QE"]); hR, hwR = ach_channel(ACH_PROC["RES"])
    nmax = 6; x = np.arange(nmax + 1)
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for row, (chan, ado, wa, ach, wh) in enumerate([("QE", aQ, wQ, hQ, hwQ), ("RES", aR, wR, hR, hwR)]):
        for ax, (lab, s, c, apid) in zip(axes[row], SPECIES):
            ax.step(x, frac(ado[lab], wa, nmax), where="mid", label="ADoNIS P=2", color="C1")
            if apid is not None:
                ax.step(x, frac(ach[lab], wh, nmax), where="mid", label="ACHILLES", color="k", lw=2)
            else:
                ax.text(0.5, 0.92, "(no ACHILLES neutrons)", transform=ax.transAxes, ha="center", fontsize=8, color="r")
            ax.set_xlabel(f"{chan}: {lab}"); ax.set_yscale("log"); ax.set_ylim(1e-4, 1.3); ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("QE: fraction of sigma"); axes[1, 0].set_ylabel("RES: fraction of sigma")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Per-channel final-state multiplicity (C): ADoNIS P=2 (full-batch) vs ACHILLES")
    fig.tight_layout(); fig.savefig(OUT, dpi=110); print(f"wrote {OUT}", flush=True)
    for row, (chan, ado, wa, ach) in enumerate([("QE", aQ, wQ, hQ), ("RES", aR, wR, hR)]):
        for lab, *_, apid in SPECIES:
            b = frac(ado[lab], wa, nmax); h = frac(ach[lab], (hwQ if chan == "QE" else hwR), nmax) if apid is not None else None
            print(f"{chan} {lab:7s} ado" + ("/ach" if h is not None else "") + " N=0..3: " +
                  " ".join(f"{b[k]:.3f}" + (f"/{h[k]:.3f}" if h is not None else "") for k in range(4)), flush=True)


if __name__ == "__main__":
    main()
