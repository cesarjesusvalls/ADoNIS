"""Hybrid ACHILLES<->ADoNIS cascade ablation: run ADoNIS's pool cascade on ACHILLES's EXACT per-event
input nucleus (parsed from an ACHILLES_CASCADEDUMP stream) and compare final-state nucleon multiplicity
+ momenta.  Because the INPUT (primary 4-momentum/position + full background config + Fermi momenta) is
identical, any difference is PURE cascade transport -- it isolates the implementation deviation behind the
QE FSI N(n) tension (mean N(n) ACH/ADO ~1.12) from input-generation differences.

Usage: python -u scripts/cascade_ablation.py <cascadedump.txt> [target=C] [--no-pauli]
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from adonis.workflow.materials import resolve_targets
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _load_density
from adonis.fsi.cascade_full import cascade_nucleus, NUCLEON
from adonis.data.oracle.parse_cascadedump import parse_cascadedump, to_arrays


def build_cfg(target, pauli=True):
    tg = resolve_targets(target)[0][0]
    ts = os.environ.get("TIMESTEP", "0") == "1"      # 1 = ACHILLES-step-matching (adaptive time-sync)
    return DiscreteCascadeConfig(step=0.04, path_budget_R=3.0, seed=1, nn_inelastic=True, pauli=pauli,
                                 time_step=ts,
                                 nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)


def su_from_achilles(arr, key):
    """Build the ADoNIS setup dict from ACHILLES's per-event nucleus (su_external).  The struck nucleon is
    ALREADY removed by ACHILLES (it is the propagating primary) -> consumed0 is all-False over the A-1
    spectators; pos0 = the primary's start position (the QE vertex)."""
    n = arr["npos"].shape[0]
    npos = jnp.asarray(arr["npos"]); nmom = jnp.asarray(arr["nmom"]); nisp = jnp.asarray(arr["nisp"])
    pos0 = jnp.asarray(arr["prim_pos"])
    consumed0 = jnp.asarray(~arr["nmask"])           # invalid (padding) slots marked consumed; real bg live
    ch0 = jnp.ones(n, jnp.int32)                     # QE: no pion (unused)
    return dict(npos=npos, nmom=nmom, nisp=nisp, pos0=pos0, consumed0=consumed0, ch0=ch0, kp=key)


def ado_finalstate(nterms):
    g0 = nterms[0]
    sp = np.asarray(g0["species"]); ch = np.asarray(g0["charge"]); al = np.asarray(g0["alive"])
    p3 = np.linalg.norm(np.asarray(g0["p4"])[:, :, 1:], axis=2)
    al = al & (p3 > 0.0)
    n_p = ((sp == NUCLEON) & (ch == 1) & al).sum(1)
    n_n = ((sp == NUCLEON) & (ch == 0) & al).sum(1)
    is_n = (sp == NUCLEON) & (ch == 0) & al
    neut_p = p3[is_n]                                          # all ejected-neutron |p| (flat)
    return n_p, n_n, neut_p


def ach_neutron_p(events):
    """ACHILLES ejected-neutron |p| (flat) from CDFS records."""
    out = []
    for e in events:
        for pid, p4 in e["fs"]:
            if pid == 2112:
                out.append(float(np.linalg.norm(p4[1:])))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("target", nargs="?", default="C")
    ap.add_argument("--no-pauli", action="store_true")
    a = ap.parse_args()
    print(f"parsing {a.dump} ...", flush=True)
    events = parse_cascadedump(a.dump)
    # QE only: events with no primary pion (proc==0)
    events = [e for e in events if e["proc"] == 0]
    arr = to_arrays(events)
    n = arr["npos"].shape[0]
    print(f"  {n} QE events; bg A={arr['A']}  mean bg p/n = {arr['nisp'].sum(1).mean():.2f}/"
          f"{(arr['nmask'].sum(1)-arr['nisp'].sum(1)).mean():.2f}", flush=True)
    cfg = build_cfg(a.target, pauli=not a.no_pauli)
    key = jax.random.PRNGKey(12345)
    su = su_from_achilles(arr, key)
    p_pi = jnp.zeros((n, 4))
    p_N = jnp.asarray(arr["prim_p4"])
    ppid = jnp.zeros(n, jnp.int32)
    ipid = jnp.full(n, 2112, jnp.int32)              # QE struck = neutron
    Npid = jnp.full(n, 2212, jnp.int32)              # outgoing proton
    print("running ADoNIS cascade on ACHILLES input ...", flush=True)
    _, nterms, _, _ = cascade_nucleus(p_pi, p_N, ppid, ipid, Npid, cfg, key, P=2, channel="qe",
                                      su_external=su)
    ado_np, ado_nn, ado_neutp = ado_finalstate(nterms)
    ach_neutp = ach_neutron_p(events)
    w = arr["w"]; W = w.sum()
    ach_np, ach_nn = arr["ach_np"], arr["ach_nn"]

    def mean(x): return float((x * w).sum() / W)
    print("\n=== IDENTICAL INPUT: ADoNIS transport vs ACHILLES transport (weighted means) ===", flush=True)
    print(f"  primary |p| match: ADoNIS==ACHILLES input (fed directly) "
          f"mean={np.linalg.norm(arr['prim_p4'][:, 1:], axis=1).mean():.1f} MeV", flush=True)
    print(f"  mean N(p): ADoNIS {mean(ado_np):.4f}   ACHILLES {mean(ach_np):.4f}   ACH/ADO {mean(ach_np)/mean(ado_np):.3f}", flush=True)
    print(f"  mean N(n): ADoNIS {mean(ado_nn):.4f}   ACHILLES {mean(ach_nn):.4f}   ACH/ADO {mean(ach_nn)/max(mean(ado_nn),1e-9):.3f}", flush=True)
    print(f"  mean secondary nucleons: ADoNIS {mean(ado_np-1)+mean(ado_nn):.4f}   ACHILLES {mean(ach_np-1)+mean(ach_nn):.4f}", flush=True)
    # N(n) DISTRIBUTION (paired, weighted) -- does the rising tension survive on identical input?
    print("\n  N(n) distribution (weighted fraction of events):", flush=True)
    print("   k     ADoNIS      ACHILLES    ACH/ADO", flush=True)
    for k in range(5):
        fa = float((w * (ado_nn == k)).sum() / W); fh = float((w * (ach_nn == k)).sum() / W)
        print(f"   {k}   {fa:.5f}    {fh:.5f}    {fh/max(fa,1e-9):.3f}", flush=True)
    # ejected-neutron |p| spectrum (UNWEIGHTED count fractions -- input is paired so this is fair)
    print("\n  ejected-neutron |p| [MeV]:", flush=True)
    print(f"   count: ADoNIS={len(ado_neutp)}  ACHILLES={len(ach_neutp)}  (ACH/ADO {len(ach_neutp)/max(len(ado_neutp),1):.3f})", flush=True)
    for q in (50, 90, 99):
        print(f"   p{q}: ADoNIS={np.percentile(ado_neutp,q):.0f}  ACHILLES={np.percentile(ach_neutp,q):.0f} MeV", flush=True)
    for thr in (300, 500, 700):
        print(f"   frac |p|>{thr}: ADoNIS={ (ado_neutp>thr).mean():.4f}  ACHILLES={ (ach_neutp>thr).mean():.4f}", flush=True)
    # ACHILLES scatter-trace stats (transport intensity it actually did)
    nsc = np.array([len(e["scat"]) for e in events]); nblk = np.array([sum(s["blocked"] for s in e["scat"]) for e in events])
    print(f"\n  ACHILLES trace: mean NN attempts/event={ (nsc*w).sum()/W:.3f}  Pauli-blocked frac={ (nblk*w).sum()/max((nsc*w).sum(),1e-9):.3f}", flush=True)
    # DIFFERENTIAL: ejected-neutron |p| spectrum on IDENTICAL input (ADoNIS vs ACHILLES) + ratio
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from adonis.workflow.plotting import chi2_ratio_panel
    edges = np.linspace(0.0, 1500.0, 31)
    _ha, _ = np.histogram(ado_neutp, edges); _hh, _ = np.histogram(ach_neutp, edges)
    print("\n  per-bin ejected-neutron |p| (UNWEIGHTED counts, paired input):", flush=True)
    print(f"  {'|p| bin':12s} {'ADO':>8s} {'ACH':>8s} {'ACH/ADO':>8s} {'pull':>7s}", flush=True)
    for i in range(len(edges) - 1):
        if _ha[i] + _hh[i] == 0:
            continue
        pull = (_hh[i] - _ha[i]) / np.sqrt(_hh[i] + _ha[i] + 1e-9)        # Poisson, paired
        print(f"  {edges[i]:5.0f}-{edges[i+1]:<5.0f} {_ha[i]:>8d} {_hh[i]:>8d} {_hh[i]/max(_ha[i],1):>8.3f} {pull:>+7.1f}", flush=True)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1], sharex=True)
    chi2_ratio_panel(a0, a1, edges, {"values": ach_neutp, "w": np.ones_like(ach_neutp)},
                     {"values": ado_neutp, "w": np.ones_like(ado_neutp)},
                     label=r"ablation: ejected-neutron $|p|$ [MeV] (identical input)",
                     ado_label="ADoNIS", ref_label="ACHILLES", logy=True)
    a0.set_ylabel("counts / bin (paired input)"); a1.set_ylabel("ACH/ADO")
    a0.legend(fontsize=8); a0.set_title("Cascade ablation (identical input): ejected-neutron |p| transport", fontsize=10)
    out = "/tmp/ablation_neutron_p.png"; fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"\n  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
