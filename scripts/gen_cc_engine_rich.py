"""HIGH-STAT reusable RICH bank from the FAITHFUL engine, shared by CC0pi (QE) and CC1pi (RES).
Per event banks everything needed to re-bin ANY signal definition off-line (no rerun):
  mu, nu, struck (4-vec) + pid_Ni ; primary pion post-FSI (4-vec)+pid ; NN-created pion (4-vec)+pid ;
  the TOP-M proton terminals (4-vec, regardless of window) ; W, Q2 ; weight ; channel (ipid/Npid).
Checkpoints after EVERY seed (saves the growing bank) so a long run is robust + the partial is usable.

Usage: python -u scripts/gen_cc_engine_rich.py <res|qe> [NRES=30000] [NSEED=56]
Output: data/oracle/t2k_{cc1pi|cc0pi}_engine_rich.npz
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec import res_xsec
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig
import adonis.fsi.cascade_full as CF

CHAN = sys.argv[1] if len(sys.argv) > 1 else "res"
NRES = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
NSEED = int(sys.argv[3]) if len(sys.argv) > 3 else 56
CFG = DiscreteCascadeConfig(cylinder=True, step=0.04, max_steps=260, seed=1, nn_inelastic=True)
MPROT = 6                                                     # top-M proton terminals stored per event
OUT = "data/oracle/t2k_%s_engine_rich.npz" % ("cc1pi" if CHAN == "res" else "cc0pi")


def gen_events(seed):
    if CHAN == "res":
        e = res_xsec.generate(NRES, seed=seed, return_events=True)["events"]
        return {k: np.asarray(e[k]) for k in ("k_nu", "k_mu", "p_struck", "p_pi", "p_N", "w", "ppid", "ipid", "Npid")}
    from scripts.gen_t2k_cc0pi_adonis import generate as qegen
    return qegen(NRES, seed=seed, return_events=True)         # qe: no p_pi (no primary pion)


def one(seed):
    a = gen_events(seed)
    n = len(a["w"]); ar = np.arange(n)
    p_pi_in = jnp.asarray(a["p_pi"]) if "p_pi" in a else jnp.asarray(a["p_N"])   # qe: dummy (ignored)
    pterm, nterms, ofl, created = CF.cascade_carbon_v2(
        p_pi_in, jnp.asarray(a["p_N"]), jnp.asarray(a["ppid"], jnp.int32), jnp.asarray(a["ipid"], jnp.int32),
        jnp.asarray(a["Npid"], jnp.int32), CFG, jax.random.PRNGKey(seed + 11), P=12, max_gen=6, channel=CHAN)
    # top-M proton terminals across all nucleon generations (regardless of window) + PROVENANCE
    p4s, orgs, gns = [], [], []
    for g in nterms:
        sp = np.asarray(g["species"]); pid = np.asarray(g["pid"]); p4 = np.asarray(g["p4"]); al = np.asarray(g["alive"])
        good = (sp == CF.NUCLEON) & (pid == 2212) & al
        p4s.append(np.where(good[:, :, None], p4, 0.0))
        orgs.append(np.where(good, np.asarray(g["origin"]), -1))      # -1 = not a (live) proton
        gns.append(np.where(good, np.asarray(g["gen"]), -1))
    P4 = np.concatenate(p4s, axis=1); ORG = np.concatenate(orgs, axis=1); GN = np.concatenate(gns, axis=1)
    mom = np.linalg.norm(P4[:, :, 1:], axis=2)
    idx = np.argsort(-mom, axis=1)[:, :MPROT]                 # top-M by momentum
    g2 = ar[:, None]
    return dict(mu=a["k_mu"], nu=a["k_nu"], struck=a["p_struck"], pid_Ni=a["ipid"].astype(np.int64),
                pi_post=np.asarray(pterm["p4"]), pid_pi=np.asarray(pterm["pid"]).astype(np.int64),
                pi_nsc=np.asarray(pterm["nsc"]).astype(np.int64),
                cr_p4=np.asarray(created["p4"]), cr_pid=np.where(np.asarray(created["alive"]),
                                                                 np.asarray(created["pid"]), 0).astype(np.int64),
                prot=P4[g2, idx], prot_origin=ORG[g2, idx].astype(np.int64), prot_gen=GN[g2, idx].astype(np.int64),
                w=a["w"] / NSEED, ipid=a["ipid"].astype(np.int64), Npid=a["Npid"].astype(np.int64))


def main():
    parts = []
    t0 = time.time()
    for sd in range(NSEED):
        parts.append(one(sd))
        out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(OUT, **out)                                  # checkpoint every seed
        sig = ((out["pid_pi"] == 211).sum() if CHAN == "res" else 0)
        print(f"  [{CHAN}] seed {sd+1}/{NSEED}  banked {len(out['w'])} ev  ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"DONE {CHAN}: {len(out['w'])} events -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
