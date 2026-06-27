"""Hybrid ACHILLES<->ADoNIS cascade ablation harness (identical-input transport comparison).

Runs ADoNIS's pool cascade on ACHILLES's EXACT per-event input nucleus (parsed from an
ACHILLES_CASCADEDUMP `.cdump` stream via su_external) so any difference is PURE cascade transport,
isolated from input-generation differences.  Consolidates the former scripts/cascade_ablation,
cascade_ablation_res, cascade_seq_test, _scatter_kin_compare into one mode-dispatched tool.

Modes:
  (default)        QE final-state N(p)/N(n) + ejected-neutron |p| spectrum
  --channel res    RES leading-neutron |p| spectrum (pion + recoil primaries)
  --seq            transport INTENSITY: successful NN scatters/event + per-generation sequencing
  --kin            NN 2->2 recoil kinematics on ACHILLES's exact per-scatter inputs (CDSCAT)

Usage:
  python -u -m analysis.validations.cascade_ablation <dump.cdump> [target=C]
      [--channel qe|res] [--seq] [--kin] [--no-pauli] [--no-inel] [--nmax N] [--logcap L] [--step fm]
Env: TIMESTEP=1 -> ACHILLES adaptive time-sync stepping (qe/seq).  Figures -> output/figures/.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root (analysis/validations/ -> .)
from adonis.workflow.materials import resolve_targets
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _load_density
from adonis.fsi.cascade_full import cascade_nucleus, NUCLEON
from analysis.validations.parse_cascadedump import parse_cascadedump, to_arrays

OUTDIR = "output/figures"
PI_CH = {211: 0, 111: 1, -211: 2}
_RES_EDGES = np.array([0, 137, 150, 200, 300, 400, 500, 700, 1000, 1500.0])


def _tg(target):
    return resolve_targets(target)[0][0]


def _su_qe(arr, key):
    """su_external for QE/seq: struck nucleon already removed by ACHILLES (it is the propagating primary)."""
    n = arr["npos"].shape[0]
    return dict(npos=jnp.asarray(arr["npos"]), nmom=jnp.asarray(arr["nmom"]), nisp=jnp.asarray(arr["nisp"]),
                pos0=jnp.asarray(arr["prim_pos"]), consumed0=jnp.asarray(~arr["nmask"]),
                ch0=jnp.ones(n, jnp.int32), kp=key)


# --------------------------------------------------------------------------- QE final state
def mode_qe(dump, target, pauli):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from adonis.workflow.plotting import chi2_ratio_panel
    print(f"parsing {dump} ...", flush=True)
    events = [e for e in parse_cascadedump(dump) if e["proc"] == 0]      # QE only
    arr = to_arrays(events); n = arr["npos"].shape[0]
    print(f"  {n} QE events; bg A={arr['A']}  mean bg p/n = {arr['nisp'].sum(1).mean():.2f}/"
          f"{(arr['nmask'].sum(1)-arr['nisp'].sum(1)).mean():.2f}", flush=True)
    tg = _tg(target); ts = os.environ.get("TIMESTEP", "0") == "1"
    cfg = DiscreteCascadeConfig(step=0.04, path_budget_R=3.0, seed=1, nn_inelastic=True, pauli=pauli,
                                time_step=ts, nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
    key = jax.random.PRNGKey(12345); su = _su_qe(arr, key)
    p_pi = jnp.zeros((n, 4)); p_N = jnp.asarray(arr["prim_p4"])
    ppid = jnp.zeros(n, jnp.int32); ipid = jnp.full(n, 2112, jnp.int32); Npid = jnp.full(n, 2212, jnp.int32)
    print("running ADoNIS cascade on ACHILLES input ...", flush=True)
    _, nterms, _, _ = cascade_nucleus(p_pi, p_N, ppid, ipid, Npid, cfg, key, channel="qe", su_external=su)
    g0 = nterms[0]
    sp = np.asarray(g0["species"]); ch = np.asarray(g0["charge"]); al = np.asarray(g0["alive"])
    p3 = np.linalg.norm(np.asarray(g0["p4"])[:, :, 1:], axis=2); al = al & (p3 > 0.0)
    ado_np = ((sp == NUCLEON) & (ch == 1) & al).sum(1); ado_nn = ((sp == NUCLEON) & (ch == 0) & al).sum(1)
    ado_neutp = p3[(sp == NUCLEON) & (ch == 0) & al]
    ach_neutp = np.array([float(np.linalg.norm(p4[1:])) for e in events for pid, p4 in e["fs"] if pid == 2112])
    w = arr["w"]; W = w.sum(); ach_np, ach_nn = arr["ach_np"], arr["ach_nn"]

    def mean(x): return float((x * w).sum() / W)
    print("\n=== IDENTICAL INPUT: ADoNIS vs ACHILLES transport (weighted means) ===", flush=True)
    print(f"  mean N(p): ADoNIS {mean(ado_np):.4f}   ACHILLES {mean(ach_np):.4f}   ACH/ADO {mean(ach_np)/mean(ado_np):.3f}", flush=True)
    print(f"  mean N(n): ADoNIS {mean(ado_nn):.4f}   ACHILLES {mean(ach_nn):.4f}   ACH/ADO {mean(ach_nn)/max(mean(ado_nn),1e-9):.3f}", flush=True)
    print("\n  N(n) distribution (weighted fraction):\n   k     ADoNIS      ACHILLES    ACH/ADO", flush=True)
    for k in range(5):
        fa = float((w * (ado_nn == k)).sum() / W); fh = float((w * (ach_nn == k)).sum() / W)
        print(f"   {k}   {fa:.5f}    {fh:.5f}    {fh/max(fa,1e-9):.3f}", flush=True)
    print(f"\n  ejected-neutron |p|: count ADoNIS={len(ado_neutp)} ACHILLES={len(ach_neutp)}", flush=True)
    for q in (50, 90, 99):
        print(f"   p{q}: ADoNIS={np.percentile(ado_neutp,q):.0f}  ACHILLES={np.percentile(ach_neutp,q):.0f} MeV", flush=True)
    nsc = np.array([len(e["scat"]) for e in events]); nblk = np.array([sum(s["blocked"] for s in e["scat"]) for e in events])
    print(f"  ACHILLES trace: NN attempts/event={(nsc*w).sum()/W:.3f}  Pauli-blocked frac={(nblk*w).sum()/max((nsc*w).sum(),1e-9):.3f}", flush=True)
    edges = np.linspace(0.0, 1500.0, 31)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1], sharex=True)
    chi2_ratio_panel(a0, a1, edges, {"values": ach_neutp, "w": np.ones_like(ach_neutp)},
                     {"values": ado_neutp, "w": np.ones_like(ado_neutp)},
                     label=r"ablation: ejected-neutron $|p|$ [MeV] (identical input)",
                     ado_label="ADoNIS", ref_label="ACHILLES", logy=True)
    a0.set_ylabel("counts / bin (paired)"); a1.set_ylabel("ACH/ADO"); a0.legend(fontsize=8)
    a0.set_title("QE cascade ablation (identical input): ejected-neutron |p|", fontsize=10)
    os.makedirs(OUTDIR, exist_ok=True); o = f"{OUTDIR}/ablation_qe_neutron_p.png"
    fig.tight_layout(); fig.savefig(o, dpi=120); print(f"\n  wrote {o}", flush=True)


# --------------------------------------------------------------------------- RES leading neutron
def _res_arrays(events, A_pad=None):
    evs = [e for e in events if e["proc"] == 1
           and any(abs(p[0]) in (211, 111) for p in e["prim"]) and any(abs(p[0]) in (2212, 2112) for p in e["prim"])]
    n = len(evs); A = A_pad or max(len(e["bg"]) for e in evs)
    p_pi = np.zeros((n, 4)); p_N = np.zeros((n, 4)); pich = np.zeros(n, np.int32); Npid = np.zeros(n, np.int64)
    pos0 = np.zeros((n, 3)); npos = np.zeros((n, A, 3)); nmom = np.zeros((n, A, 4)); nmom[:, :, 0] = 939.0
    nisp = np.zeros((n, A), bool); nmask = np.zeros((n, A), bool); w = np.zeros(n); ach_leadn = np.full(n, -1.0)
    for i, e in enumerate(evs):
        pis = [p for p in e["prim"] if abs(p[0]) in (211, 111)]; nus = [p for p in e["prim"] if abs(p[0]) in (2212, 2112)]
        pp = max(pis, key=lambda p: p[1][0]); nn = max(nus, key=lambda p: p[1][0])
        p_pi[i] = pp[1]; pich[i] = PI_CH[pp[0]]; pos0[i] = nn[2]; p_N[i] = nn[1]; Npid[i] = nn[0]
        for a, (idx, pid, p4, pos) in enumerate(e["bg"][:A]):
            npos[i, a] = pos; nmom[i, a] = p4; nisp[i, a] = (pid == 2212); nmask[i, a] = True
        nps = [float(np.linalg.norm(p4[1:])) for pid, p4 in e["fs"] if pid == 2112]
        if nps:
            ach_leadn[i] = max(nps)
        w[i] = e["w"]
    return dict(p_pi=p_pi, p_N=p_N, pich=pich, Npid=Npid, pos0=pos0, npos=npos, nmom=nmom,
                nisp=nisp, nmask=nmask, w=w, ach_leadn=ach_leadn, A=A, n=n)


def mode_res(dump, target, pauli, inel):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from adonis.workflow.plotting import chi2_ratio_panel
    print(f"parsing {dump} ...", flush=True)
    arr = _res_arrays(parse_cascadedump(dump)); n = arr["n"]
    print(f"  {n} RES events; bg A={arr['A']}  pion charge frac +/0/-={np.bincount(arr['pich'], minlength=3)/n}", flush=True)
    tg = _tg(target)
    cfg = DiscreteCascadeConfig(step=0.04, max_steps=100000, path_budget_R=3.0, seed=1, nn_inelastic=inel,
                                pauli=pauli, nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
    key = jax.random.PRNGKey(12345)
    su = dict(npos=jnp.asarray(arr["npos"]), nmom=jnp.asarray(arr["nmom"]), nisp=jnp.asarray(arr["nisp"]),
              pos0=jnp.asarray(arr["pos0"]), consumed0=jnp.asarray(~arr["nmask"]),
              ch0=jnp.asarray(arr["pich"]), kp=key)
    p_pi = jnp.asarray(arr["p_pi"]); p_N = jnp.asarray(arr["p_N"])
    ppid = jnp.full(n, 211, jnp.int32); ipid = jnp.full(n, 2112, jnp.int32); Npid = jnp.asarray(arr["Npid"]).astype(jnp.int32)
    print(f"running ADoNIS pool RES cascade on ACHILLES input (pauli={pauli} inel={inel}) ...", flush=True)
    nterms = cascade_nucleus(p_pi, p_N, ppid, ipid, Npid, cfg, key, channel="res", su_external=su)[1]
    g0 = nterms[0]; sp = np.asarray(g0["species"]); ch = np.asarray(g0["charge"]); al = np.asarray(g0["alive"])
    p3 = np.linalg.norm(np.asarray(g0["p4"])[:, :, 1:], axis=2); isn = (sp == NUCLEON) & (ch == 0) & al & (p3 > 0)
    ado = np.where(isn, p3, -1.0).max(1); ach = arr["ach_leadn"]; w = arr["w"]

    def wbin(vals):
        m = vals >= 0; h, _ = np.histogram(vals[m], _RES_EDGES, weights=w[m]); h2, _ = np.histogram(vals[m], _RES_EDGES, weights=w[m] ** 2)
        return h, np.sqrt(h2)
    da, ea = wbin(ado); dh, eh = wbin(ach)
    print("\n=== IDENTICAL INPUT: LEADING-neutron |p| (weighted) ===", flush=True)
    print(f"  {'|p| bin':12s} {'ADO':>11s} {'ACH':>11s} {'ACH/ADO':>8s} {'pull':>7s}", flush=True)
    for k in range(len(_RES_EDGES) - 1):
        pull = (dh[k] - da[k]) / np.sqrt(ea[k] ** 2 + eh[k] ** 2 + 1e-300)
        print(f"  {_RES_EDGES[k]:5.0f}-{_RES_EDGES[k+1]:<5.0f}  {da[k]:.4e} {dh[k]:.4e} {dh[k]/max(da[k],1e-30):8.3f} {pull:+7.2f}", flush=True)
    print(f"  soft(<150) ACH/ADO={ (dh[0]+dh[1])/max(da[0]+da[1],1e-30):.3f}   total ACH/ADO={dh.sum()/max(da.sum(),1e-30):.3f}", flush=True)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1], sharex=True)
    ma = ado >= 0; mh = ach >= 0
    chi2_ratio_panel(a0, a1, _RES_EDGES, {"values": ach[mh], "w": w[mh]}, {"values": ado[ma], "w": w[ma]},
                     label="ablation: leading-neutron |p| [MeV] (identical RES input)",
                     ado_label="ADoNIS", ref_label="ACHILLES", logy=True)
    a0.set_ylabel("dsigma/bin (paired)"); a1.set_ylabel("ACH/ADO"); a0.legend(fontsize=8)
    a0.set_title(f"RES cascade ablation (identical input): leading-neutron |p| (pauli={pauli} inel={inel})", fontsize=9)
    os.makedirs(OUTDIR, exist_ok=True); o = f"{OUTDIR}/ablation_res_leadn.png"
    fig.tight_layout(); fig.savefig(o, dpi=120); print(f"\n  wrote {o}", flush=True)


# --------------------------------------------------------------------------- sequencing intensity
def mode_seq(dump, target, nmax, logcap, step):
    timestep = os.environ.get("TIMESTEP", "0") == "1"
    print(f"parsing {dump} (NMAX={nmax}) ...", flush=True)
    events = [e for e in parse_cascadedump(dump) if e["proc"] == 0][:nmax]
    arr = to_arrays(events); n = arr["npos"].shape[0]; w = arr["w"]; W = w.sum()
    print(f"  {n} QE events  stepping={'TIME-sync' if timestep else 'DISTANCE-sync'}  step={step} fm", flush=True)
    ach_succ = np.zeros(n); ach_att = np.zeros(n); ach_blk = np.zeros(n); ach_succ_nrec = np.zeros(n)
    for i, e in enumerate(events):
        for s in e["scat"]:
            ach_att[i] += 1
            if s["blocked"]:
                ach_blk[i] += 1; continue
            ach_succ[i] += 1
            if s["pid2"] == 2112:
                ach_succ_nrec[i] += 1
    tg = _tg(target)
    cfg = DiscreteCascadeConfig(step=step, path_budget_R=3.0, seed=1, nn_inelastic=True, pauli=True,
                                time_step=timestep, nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
    key = jax.random.PRNGKey(12345); su = _su_qe(arr, key)
    p_pi = jnp.zeros((n, 4)); p_N = jnp.asarray(arr["prim_p4"])
    ppid = jnp.zeros(n, jnp.int32); ipid = jnp.full(n, 2112, jnp.int32); Npid = jnp.full(n, 2212, jnp.int32)
    print(f"running ADoNIS cascade w/ segment logger (LOGCAP={logcap}) ...", flush=True)
    log, counts, ovf = cascade_nucleus(p_pi, p_N, ppid, ipid, Npid, cfg, key, channel="qe",
                                       su_external=su, log_cap=logcap)
    counts = np.asarray(counts); chan = np.asarray(log["chan"]); inc_pid = np.asarray(log["inc_pid"]); gen = np.asarray(log["gen"])
    n1_pid = np.asarray(log["n1_pid"]); n1_al = np.asarray(log["n1_al"]); n2_pid = np.asarray(log["n2_pid"]); n2_al = np.asarray(log["n2_al"])
    rowmask = np.arange(chan.shape[1])[None, :] < counts[:, None]
    succ = (np.isin(inc_pid, (2212, 2112)) & rowmask) & np.isin(chan, (1, 2))
    ado_succ = succ.sum(1).astype(float)
    eject_n = succ & (((n1_pid == 2112) & (n1_al > 0)) | ((n2_pid == 2112) & (n2_al > 0)))
    ado_succ_nrec = eject_n.sum(1).astype(float)
    ado_g0 = (succ & (gen == 0)).sum(1).astype(float); ado_g1 = (succ & (gen >= 1)).sum(1).astype(float)

    def m(x): return float((x * w).sum() / W)
    print("\n=== CASCADE INTENSITY on IDENTICAL input (weighted means/event) ===", flush=True)
    print(f"  successful NN scatters:      ADoNIS {m(ado_succ):.4f}   ACHILLES {m(ach_succ):.4f}   ACH/ADO {m(ach_succ)/max(m(ado_succ),1e-9):.3f}", flush=True)
    print(f"  scatters ejecting a neutron: ADoNIS {m(ado_succ_nrec):.4f}   ACHILLES {m(ach_succ_nrec):.4f}   ACH/ADO {m(ach_succ_nrec)/max(m(ado_succ_nrec),1e-9):.3f}", flush=True)
    print(f"  ACHILLES attempts (incl blkd): {m(ach_att):.4f}   blocked frac {m(ach_blk)/max(m(ach_att),1e-9):.3f}", flush=True)
    print(f"  ADoNIS successful by GEN: gen0={m(ado_g0):.4f}  gen>=1={m(ado_g1):.4f}", flush=True)
    print("\n  successful-NN-scatters/event distribution (weighted fraction):\n   k     ADoNIS      ACHILLES    ACH/ADO", flush=True)
    for k in range(6):
        fa = float((w * (ado_succ == k)).sum() / W); fh = float((w * (ach_succ == k)).sum() / W)
        print(f"   {k}   {fa:.5f}    {fh:.5f}    {fh/max(fa,1e-9):.3f}", flush=True)


# --------------------------------------------------------------------------- 2->2 recoil kinematics
def mode_kin(dump, nmax):
    from adonis.fsi.cascade_real import _two_body_cm_scatter
    from adonis.constants import mp as MP, mn as MN

    def vec(s): return [float(x) for x in s.split(",")]
    in1 = []; in2 = []; ach_rec = []; nsc = 0
    with open(dump) as f:
        for line in f:
            if not line.startswith("CDSCAT"):
                continue
            d = {}; outs = []
            for t in line.split()[1:]:
                if t.startswith("out="):
                    p = t[4:].split(","); outs.append((int(p[0]), [float(x) for x in p[1:]]))
                elif "=" in t:
                    k, v = t.split("=", 1); d[k] = v
            if int(d["type"]) != 1 or int(d["pid2"]) != 2112:
                continue
            rec = [o for o in outs if o[0] == 2112]
            if not rec:
                continue
            in1.append(vec(d["in1"])); in2.append(vec(d["in2"])); ach_rec.append(rec[0][1]); nsc += 1
            if nsc >= nmax:
                break
    in1 = np.array(in1); in2 = np.array(in2); ach_p = np.linalg.norm(np.array(ach_rec)[:, 1:], axis=1)
    n = len(in1); keys = jax.random.split(jax.random.PRNGKey(0), n)

    def scat(l, s, k):
        po = _two_body_cm_scatter(l, s, MP, k, m_recoil=MN); r = (l + s) - po
        return jnp.linalg.norm(r[1:])
    ado_p = np.array(jax.vmap(scat)(jnp.array(in1), jnp.array(in2), keys))
    print(f"pn scatters compared: {n}", flush=True)
    edges = np.array([0, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 1000, 1500.])
    ha, _ = np.histogram(ado_p, edges); hh, _ = np.histogram(ach_p, edges)
    print(f"  {'recoil-n |p|':14s} {'ADO':>8s} {'ACH':>8s} {'ACH/ADO':>8s}", flush=True)
    for i in range(len(edges) - 1):
        print(f"  {edges[i]:5.0f}-{edges[i+1]:<5.0f} {ha[i]:>8d} {hh[i]:>8d} {hh[i]/max(ha[i],1):>8.3f}", flush=True)
    print(f"  median |p|: ADO={np.median(ado_p):.0f}  ACH={np.median(ach_p):.0f} MeV", flush=True)
    print(f"  mean |p|: ADO={ado_p.mean():.1f}  ACH={ach_p.mean():.1f}  std ADO={ado_p.std():.1f} ACH={ach_p.std():.1f}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="ACHILLES<->ADoNIS cascade ablation (identical-input transport).")
    ap.add_argument("dump")
    ap.add_argument("target", nargs="?", default="C")
    ap.add_argument("--channel", default="qe", choices=["qe", "res"])
    ap.add_argument("--seq", action="store_true", help="sequencing/intensity mode (QE)")
    ap.add_argument("--kin", action="store_true", help="NN 2->2 recoil kinematics mode")
    ap.add_argument("--no-pauli", action="store_true")
    ap.add_argument("--no-inel", action="store_true")
    ap.add_argument("--nmax", type=int, default=200000)
    ap.add_argument("--logcap", type=int, default=96)
    ap.add_argument("--step", type=float, default=0.04)
    a = ap.parse_args(argv)
    if a.kin:
        mode_kin(a.dump, a.nmax)
    elif a.seq:
        mode_seq(a.dump, a.target, a.nmax, a.logcap, a.step)
    elif a.channel == "res":
        mode_res(a.dump, a.target, not a.no_pauli, not a.no_inel)
    else:
        mode_qe(a.dump, a.target, not a.no_pauli)


if __name__ == "__main__":
    main()
