"""Single-harness cut-by-cut bisect of the CC1pi RES-C signal fraction: ADO event-kernel vs
ACHILLES, with IDENTICAL cut definitions and denominators (raw event counts; no SCALE).
Cuts applied cumulatively: mu -> exactly-one-pi+ -> pi-window -> in-window proton.
Usage: python scripts/cc1pi_cut_bisect.py [N_per_seed=150000] [NSEED=4]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec import res_xsec
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig
from adonis.fsi.cascade_event import evolve_event
from adonis.data.oracle.parse_hepmc import parse_events

C70 = np.cos(np.deg2rad(70.0))
CUTS = ["all", "+mu", "+1pi+", "+piwin", "+prot"]


def ach_counts(path, nmax=600000):
    c = np.zeros(5)
    n_res = 0
    for evt in parse_events(Path(path)):
        if (evt.get("proc") or 0) not in (401, 402):
            continue
        pstr = None; mu = None; pis = []; prot_ok = False
        for pid, status, p4 in evt["parts"]:
            if status == 2 and pid in (2112, 2212) and pstr is None:
                pstr = p4
            if status != 1:
                continue
            if pid == 13:
                mu = np.asarray(p4)
            elif pid in (211, 111, -211, 221, 130, 310, 311, 321, -321, -311):
                pis.append((pid, np.asarray(p4)))
            elif pid == 2212:
                pm = np.linalg.norm(np.asarray(p4)[1:])
                if 450 < pm < 1200 and p4[3] / max(pm, 1e-9) > C70:
                    prot_ok = True
        if pstr is None or (pstr[1]**2 + pstr[2]**2 + pstr[3]**2) ** 0.5 < 1.0:
            continue                                   # carbon only
        n_res += 1
        c[0] += 1
        mu_ok = mu is not None and 250 < np.linalg.norm(mu[1:]) < 7000 \
            and mu[3] / max(np.linalg.norm(mu[1:]), 1e-9) > C70
        if not mu_ok: continue
        c[1] += 1
        if not (len(pis) == 1 and pis[0][0] == 211): continue
        c[2] += 1
        pp = pis[0][1]; pm = np.linalg.norm(pp[1:])
        if not (150 < pm < 1200 and pp[3] / max(pm, 1e-9) > C70): continue
        c[3] += 1
        if prot_ok: c[4] += 1
        if n_res >= nmax: break
    return c


def ado_counts(n, nseed):
    cfg = DiscreteCascadeConfig(cylinder=True, step=0.04, max_steps=300)
    c = np.zeros(5)
    for sd in range(nseed):
        e = res_xsec.generate(n, seed=sd, return_events=True)["events"]
        w = np.asarray(e["w"])
        kmu = np.asarray(e["k_mu"]); mm = np.linalg.norm(kmu[:, 1:], axis=1)
        mu_ok = (mm > 250) & (mm < 7000) & (kmu[:, 3] / np.clip(mm, 1e-9, None) > C70)
        out = evolve_event(jnp.asarray(e["p_pi"]), jnp.asarray(e["p_N"]),
                           jnp.asarray(e["ppid"], jnp.int32), jnp.asarray(e["ipid"], jnp.int32),
                           jnp.asarray(e["Npid"], jnp.int32), cfg, jax.random.PRNGKey(sd * 13 + 3))
        spec = np.asarray(out["spec"]); chg = np.asarray(out["chg"]); pf = np.asarray(out["p"])
        pm = np.linalg.norm(pf[..., 1:], axis=-1); cth = pf[..., 3] / np.clip(pm, 1e-9, None)
        is_pi = spec == 1
        one_pip = (is_pi.sum(axis=1) == 1) & ((is_pi & (chg == 1)).sum(axis=1) == 1)
        piwin = (is_pi & (chg == 1) & (pm > 150) & (pm < 1200) & (cth > C70)).any(axis=1)
        prot = ((spec == 2) & (chg == 1) & (pm > 450) & (pm < 1200) & (cth > C70)).any(axis=1)
        masks = [np.ones(len(w), bool), mu_ok, mu_ok & one_pip, mu_ok & one_pip & piwin,
                 mu_ok & one_pip & piwin & prot]
        for i, msk in enumerate(masks):
            c[i] += w[msk].sum()
        print(f"  ado seed {sd+1}/{nseed}", flush=True)
    return c


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 150000
    NS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ca = ach_counts("_oracle_out/T2K_CH_virt.hepmc")
    print("ACH cumulative fractions:", " ".join(f"{nm}={x/ca[0]:.4f}" for nm, x in zip(CUTS, ca)), flush=True)
    cd = ado_counts(N, NS)
    print("ADO cumulative fractions:", " ".join(f"{nm}={x/cd[0]:.4f}" for nm, x in zip(CUTS, cd)), flush=True)
    print("step survival ratios ADO/ACH:",
          " ".join(f"{CUTS[i]}:{(cd[i]/cd[i-1])/(ca[i]/ca[i-1]):.4f}" for i in range(1, 5)), flush=True)
