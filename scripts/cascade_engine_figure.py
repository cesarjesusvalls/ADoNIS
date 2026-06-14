"""The usual CC1pi+Np diff-xsec + ACH/ADO ratio figure, but ADoNIS = the FAITHFUL BFS cascade engine
(adonis/fsi/cascade_full) instead of the factorized chain.  Runs the engine over NSEED seeds, extracts
the full signal (surviving pi+ = gen-0 pion; leading in-window proton across ALL generations; muon from
the event), computes the standard observables, and overlays vs the ACHILLES rich bank (fixed sigdef).

Usage: python -u scripts/cascade_engine_figure.py [NRES=30000] [NSEED=4]
Output: paper_figures/cc1pi_ratios_ENGINE_carbon.png ; caches data/oracle/t2k_cc1pi_engine_sig.npz
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec import res_xsec
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig
import adonis.fsi.cascade_full as CF
import scripts.cc1pi_fig_tki as F            # observables() (single source)
import scripts.cc1pi_signal as S
from adonis.data.oracle.normalization import hepmc_norm

NRES = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 4
COS70 = np.cos(np.deg2rad(70.0)); P, MG = 10, 6
CFG = DiscreteCascadeConfig(cylinder=True, step=0.04, max_steps=260, seed=1, nn_inelastic=True)
MU_LO, MU_HI, PI_LO, PI_HI, P_LO, P_HI = 250., 7000., 150., 1200., 450., 1200.
# Fixed batch shape across seeds -> compiles ONCE (no per-seed re-JIT).  We generate ALL seeds first,
# set N_PAD = the MAX accepted-event count across them, and pad shorter seeds with w=0 dummies (excluded
# by the w>0 cut).  No event is ever dropped (no truncation) and there is no per-seed re-JIT.
_KEYS = ("k_nu", "k_mu", "p_struck", "p_pi", "p_N", "w", "ppid", "ipid", "Npid")


def _pad_to(a, N_PAD):
    m = len(a["w"])
    if m == N_PAD:
        return a
    pad = N_PAD - m
    out = {k: np.concatenate([v, np.broadcast_to(v[:1], (pad,) + v.shape[1:])], axis=0) for k, v in a.items()}
    out["w"] = out["w"].copy(); out["w"][m:] = 0.0
    return out


def _acc(p4, lo, hi):
    m = np.linalg.norm(p4[:, 1:], axis=1)
    return (m > lo) & (m < hi) & (p4[:, 3] / np.clip(m, 1e-9, None) > COS70)


def engine_signal():
    cells = []
    # generate ALL seeds first, then pad every seed to the MAX count -> one compile, no truncation
    raw = []
    for sd in range(NSEED):
        e = res_xsec.generate(NRES, seed=sd, return_events=True)["events"]
        raw.append({k: np.asarray(e[k]) for k in _KEYS})
        print(f"  gen seed {sd}: {len(raw[-1]['w'])} events", flush=True)
    N_PAD = max(len(a["w"]) for a in raw)
    print(f"  N_PAD = max over seeds = {N_PAD} (single compile, no dropped events)", flush=True)
    for sd in range(NSEED):
        a = _pad_to(raw[sd], N_PAD)
        knu, kmu, pstr = a["k_nu"], a["k_mu"], a["p_struck"]
        ppi = jnp.asarray(a["p_pi"]); pN = jnp.asarray(a["p_N"]); w = a["w"]
        ppid = jnp.asarray(a["ppid"], jnp.int32); ipid = jnp.asarray(a["ipid"], jnp.int32); Npid = jnp.asarray(a["Npid"], jnp.int32)
        pterm, nterms, ofl = CF.cascade_carbon_v2(ppi, pN, ppid, ipid, Npid, CFG, jax.random.PRNGKey(sd + 11), P=12, max_gen=MG)
        n = len(w); ar = np.arange(n)
        # pion terminal (surviving pi+)
        pi_f = np.asarray(pterm["p4"]); pid_pi = np.asarray(pterm["pid"])
        # leading in-window proton across ALL nucleon generations
        best = np.zeros((n, 4)); bm = np.zeros(n)
        for g in nterms:
            sp = np.asarray(g["species"]); pid = np.asarray(g["pid"]); p4 = np.asarray(g["p4"]); al = np.asarray(g["alive"])
            pm = np.linalg.norm(p4[:, :, 1:], axis=2); cth = p4[:, :, 3] / np.clip(pm, 1e-9, None)
            mask = (sp == CF.NUCLEON) & (pid == 2212) & al & (pm > P_LO) & (pm < P_HI) & (cth > COS70)
            mm = np.where(mask, pm, -1.0); j = np.argmax(mm, axis=1); gm = mm[ar, j]
            upd = gm > bm; best = np.where(upd[:, None], p4[ar, j], best); bm = np.where(upd, gm, bm)
        has_p = bm > 0
        sel = (pid_pi == 211) & has_p & (w > 0) & _acc(kmu, MU_LO, MU_HI) & _acc(pi_f, PI_LO, PI_HI)
        dptt, pn, dat, dpt = F.observables(kmu[sel], pi_f[sel], best[sel], np.zeros(int(sel.sum()), bool), sd)
        qv = (knu - kmu)[sel]; totv = qv + pstr[sel]
        Wv = np.sqrt(np.clip(totv[:, 0] ** 2 - np.sum(totv[:, 1:] ** 2, axis=1), 0, None))
        Q2v = (np.sum(qv[:, 1:] ** 2, axis=1) - qv[:, 0] ** 2) / 1e6
        pim = np.linalg.norm(pi_f[sel][:, 1:], axis=1); lpm = np.linalg.norm(best[sel][:, 1:], axis=1)
        cells.append(dict(pn=pn, dptt=dptt, dalphat=dat, W=Wv, Q2=Q2v, pi_p=pim, lp_p=lpm, w=w[sel] / NSEED))
        print(f"  seed {sd}: signal {int(sel.sum())}  overflow {int(ofl)}", flush=True)
    A = {k: np.concatenate([c[k] for c in cells]) for k in cells[0]}
    np.savez("data/oracle/t2k_cc1pi_engine_sig.npz", **A)
    return A


VARS = [("pn", np.array([0, 120, 240, 600, 1500.]), r"$p_N$"), ("dptt", np.array([-700, -300, -100, 100, 300, 700.]), r"$\delta p_{TT}$"),
        ("dalphat", np.linspace(0, np.pi, 7), r"$\delta\alpha_T$"), ("W", np.linspace(1080, 1700, 13), r"$W$"),
        ("Q2", np.linspace(0, 1.5, 13), r"$Q^2$"), ("pi_p", np.linspace(150, 1200, 13), r"$p_\pi$"),
        ("lp_p", np.linspace(450, 1200, 13), r"lead $p_p$")]


def main():
    A = engine_signal()
    ach = dict(np.load("data/oracle/t2k_cc1pi_rich_ach_FSI.npz"))
    H = S.ach_select(ach, dict(S.DEFAULT)); H["w"] = H["w"] * float(ach["weight_to_nb"])
    print(f"\nselected sigma: ENGINE {A['w'].sum():.4e}  ACH {H['w'].sum():.4e}  ACH/ADO {H['w'].sum()/max(A['w'].sum(),1e-30):.3f}")
    print(f"{'var':8s} chi2/ndf  ACH/ADO")
    fig, ax = plt.subplots(2, len(VARS), figsize=(3.4 * len(VARS), 6.2), height_ratios=[3, 1], squeeze=False, sharex="col")
    for c, (k, edg, xl) in enumerate(VARS):
        bw = np.diff(edg); ctr = 0.5 * (edg[1:] + edg[:-1])
        da, _ = np.histogram(H[k], edg, weights=H["w"]); ea = np.sqrt(np.histogram(H[k], edg, weights=H["w"]**2)[0]) / bw; da /= bw
        dd, _ = np.histogram(A[k], edg, weights=A["w"]); ed = np.sqrt(np.histogram(A[k], edg, weights=A["w"]**2)[0]) / bw; dd /= bw
        a0, a1 = ax[0, c], ax[1, c]
        a0.fill_between(edg, np.append(da-ea, (da-ea)[-1]), np.append(da+ea, (da+ea)[-1]), step="post", color="0.55", alpha=0.55, lw=0, label="ACH stat")
        a0.step(edg, np.append(da, da[-1]), where="post", color="0.3", lw=1.3, label="ACHILLES")
        a0.errorbar(ctr, dd, yerr=ed, fmt="s", color="C0", ms=3, capsize=2, lw=0.9, label="ENGINE")
        a0.set_title(xl, fontsize=9); a0.set_ylim(bottom=0)
        if c == 0: a0.legend(fontsize=7)
        m = (da > 0) & (dd > 0); chi2 = float(np.sum((da[m]-dd[m])**2 / (ea[m]**2+ed[m]**2))); ndf = int(m.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            r = da/dd; re = r*np.sqrt((ed/dd)**2+(ea/da)**2)
        a1.axhspan(0.9, 1.1, color="green", alpha=0.12); a1.axhline(1.0, ls="--", color="green", lw=0.7)
        a1.errorbar(ctr[m], r[m], yerr=re[m], fmt="o", color="C3", ms=3, capsize=2, lw=0.8)
        a1.set_ylim(0.5, 1.6); a1.set_xlabel(xl, fontsize=8); a1.text(0.04, 0.83, f"{chi2/max(ndf,1):.1f}", transform=a1.transAxes, fontsize=9)
        print(f"{k:8s}  {chi2/max(ndf,1):6.2f}   {np.sum(da*bw)/max(np.sum(dd*bw),1e-30):.3f}")
    fig.suptitle(f"CC1$\\pi^+$Np  FAITHFUL ENGINE vs ACHILLES (carbon)  ACH/ADO {H['w'].sum()/max(A['w'].sum(),1e-30):.3f}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); out = "paper_figures/cc1pi_ratios_ENGINE_carbon.png"; fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
