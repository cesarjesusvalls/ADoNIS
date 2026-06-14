"""P4-lite: does the faithful BFS engine's deeper shared-nucleus re-cascade move the proton-leg
acceptance P(in-window proton | pi_p) toward ACHILLES?  Engine as-is (leading secondary, no production
change).  Compares the engine to the ACHILLES rich-bank conditional (the deficit localized in
cc1pi_t2k #30-33).  No regeneration of ACHILLES.

Usage: python -u scripts/cascade_engine_protoncheck.py [NRES=60000] [NSEED=2]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec import res_xsec
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig
import adonis.fsi.cascade_full as CF
from adonis.data.oracle.normalization import hepmc_norm

NRES = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 2
COS70 = np.cos(np.deg2rad(70.0))
CFG = DiscreteCascadeConfig(cylinder=True, step=0.04, max_steps=260, seed=1, nn_inelastic=True)
Ped = np.array([150, 300, 450, 600, 750, 900, 1050, 1200.]); CTR = 0.5 * (Ped[1:] + Ped[:-1])


def mom(p): return np.linalg.norm(p[:, 1:], axis=1)


def engine_conditional():
    num = np.zeros(len(CTR)); den = np.zeros(len(CTR))
    for sd in range(NSEED):
        e = res_xsec.generate(NRES, seed=sd, return_events=True)["events"]
        ppi = jnp.asarray(e["p_pi"]); pN = jnp.asarray(e["p_N"]); w = np.asarray(e["w"])
        kmu = np.asarray(e["k_mu"])
        ppid = jnp.asarray(e["ppid"], jnp.int32); ipid = jnp.asarray(e["ipid"], jnp.int32); Npid = jnp.asarray(e["Npid"], jnp.int32)
        terms, ofl = CF.cascade_carbon(ppi, pN, ppid, ipid, Npid, CFG, jax.random.PRNGKey(sd + 11), P=10, max_gen=6)
        n = len(w)
        # gen0 pion (surviving pi+) and its momentum
        g0 = terms[0]; sp0 = np.asarray(g0["species"]); pid0 = np.asarray(g0["pid"]); p40 = np.asarray(g0["p4"])
        is_pi = sp0 == CF.PION
        pislot = np.argmax(is_pi, axis=1)                                  # the gen0 pion slot
        pip = p40[np.arange(n), pislot]; pim = mom(pip); picth = pip[:, 3] / np.clip(pim, 1e-9, None)
        pid_pi = pid0[np.arange(n), pislot]
        mu_m = np.linalg.norm(kmu[:, 1:], axis=1); mu_acc = (mu_m > 250) & (mu_m < 7000) & (kmu[:, 3] / np.clip(mu_m, 1e-9, None) > COS70)
        base = (pid_pi == 211) & (pim > 150) & (pim < 1200) & (picth > COS70) & mu_acc & (w > 0)
        # in-window proton in ANY generation/slot
        has_p = np.zeros(n, bool)
        for g in terms:
            sp = np.asarray(g["species"]); pid = np.asarray(g["pid"]); p4 = np.asarray(g["p4"]); al = np.asarray(g["alive"])
            pm = np.linalg.norm(p4[:, :, 1:], axis=2); cth = p4[:, :, 3] / np.clip(pm, 1e-9, None)
            inwin = (sp == CF.NUCLEON) & (pid == 2212) & al & (pm > 450) & (pm < 1200) & (cth > COS70)
            has_p = has_p | inwin.any(axis=1)
        for i in range(len(CTR)):
            m = base & (pim >= Ped[i]) & (pim < Ped[i + 1])
            num[i] += w[m & has_p].sum(); den[i] += w[m].sum()
        print(f"  seed {sd}: base {int(base.sum())}  overflow {int(ofl)}", flush=True)
    return np.where(den > 0, num / den, np.nan)


def ach_conditional():
    a = np.load("data/oracle/t2k_res_w_achilles_FSI.npz")
    w = a["w"].astype(float) * hepmc_norm("_oracle_out/T2K_CH_virt.hepmc")["weight_to_nb"]
    base = ((a["pstr"] > 1) & (a["mu_p"] > 250) & (a["mu_p"] < 7000) & (a["mu_cth"] > COS70)
            & (a["pi_pid"] == 211) & (a["pi_p"] > 150) & (a["pi_p"] < 1200) & (a["pi_cth"] > COS70) & (a["n_pi"] == 1))
    pip = a["pi_p"]; pok = a["prot_ok"].astype(bool)
    out = []
    for i in range(len(CTR)):
        m = base & (pip >= Ped[i]) & (pip < Ped[i + 1]); out.append(w[m & pok].sum() / max(w[m].sum(), 1e-30))
    return np.array(out)


def main():
    eng = engine_conditional(); ach = ach_conditional()
    print(f"\n{'pi_p':>6} {'ACH':>7} {'ENGINE':>7} {'ENG/ACH':>8}   (P(in-window proton | pi_p), carbon)")
    for i in range(len(CTR)):
        print(f"{CTR[i]:6.0f} {ach[i]:7.3f} {eng[i]:7.3f} {eng[i]/max(ach[i],1e-9):8.3f}")
    print("\nReference (current factorized chain, after recoil-pid fix): ENG/ACH was 0.88(lo)->0.70(hi) -- too few.")


if __name__ == "__main__":
    main()
