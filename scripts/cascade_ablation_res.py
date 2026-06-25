"""RES cascade ablation: run ADoNIS's pool cascade on ACHILLES's EXACT per-event RES final state +
nucleus (parsed from an ACHILLES_CASCADEDUMP stream, proc==1) and compare the per-event LEADING-neutron
|p| spectrum.  Because the INPUT (primary pion + recoil nucleon 4-momenta/position + full background
config + Fermi momenta) is identical, any difference is PURE cascade transport -- it isolates the
implementation deviation behind the RES rank-0 (leading) neutron soft-excess (ADoNIS ~+13% at <150 MeV)
from input-generation differences.

The pool gen-0 stack for RES = [primary pion, recoil nucleon] at the RES vertex (cascade_full._cascade_pool);
ch0 = pion charge idx (0:pi+,1:pi0,2:pi-), Npid = recoil nucleon pid.  su_external bypasses our sampling.

Usage: python -u scripts/cascade_ablation_res.py <dump.cdump> [target=C] [--no-pauli] [--no-inel]
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from adonis.workflow.materials import resolve_targets
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _load_density
from adonis.fsi.cascade_full import cascade_nucleus, NUCLEON
from adonis.data.oracle.parse_cascadedump import parse_cascadedump

PI_CH = {211: 0, 111: 1, -211: 2}                       # pion pid -> charge index (0:+,1:0,2:-)
EDGES = np.array([0, 137, 150, 200, 300, 400, 500, 700, 1000, 1500.0])


def build_cfg(target, pauli=True, inel=True):
    tg = resolve_targets(target)[0][0]
    _, _, _, r = _load_density(tg.density_p, tg.density_n)
    ms = max(600, int(np.ceil(3.0 * r / 0.04)))
    return DiscreteCascadeConfig(step=0.04, max_steps=ms, seed=1, nn_inelastic=inel, pauli=pauli,
                                 nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)


def res_arrays(events, A_pad=None):
    """Pack RES (proc==1) events with a pion primary + recoil nucleon primary into su_external arrays +
    the ACHILLES per-event leading-neutron |p| (max over final-state neutrons; -1 if none)."""
    evs = []
    for e in events:
        if e["proc"] != 1:
            continue
        pis = [p for p in e["prim"] if abs(p[0]) in (211, 111)]
        nus = [p for p in e["prim"] if abs(p[0]) in (2212, 2112)]
        if pis and nus:
            evs.append(e)
    n = len(evs)
    A = A_pad or max(len(e["bg"]) for e in evs)
    p_pi = np.zeros((n, 4)); p_N = np.zeros((n, 4)); pich = np.zeros(n, np.int32); Npid = np.zeros(n, np.int64)
    pos0 = np.zeros((n, 3))
    npos = np.zeros((n, A, 3)); nmom = np.zeros((n, A, 4)); nmom[:, :, 0] = 939.0
    nisp = np.zeros((n, A), bool); nmask = np.zeros((n, A), bool); w = np.zeros(n)
    ach_leadn = np.full(n, -1.0)
    for i, e in enumerate(evs):
        pis = [p for p in e["prim"] if abs(p[0]) in (211, 111)]
        nus = [p for p in e["prim"] if abs(p[0]) in (2212, 2112)]
        pp = max(pis, key=lambda p: p[1][0]); nn = max(nus, key=lambda p: p[1][0])
        p_pi[i] = pp[1]; pich[i] = PI_CH[pp[0]]; pos0[i] = nn[2]
        p_N[i] = nn[1]; Npid[i] = nn[0]
        for a, (idx, pid, p4, pos) in enumerate(e["bg"][:A]):
            npos[i, a] = pos; nmom[i, a] = p4; nisp[i, a] = (pid == 2212); nmask[i, a] = True
        nps = [float(np.linalg.norm(p4[1:])) for pid, p4 in e["fs"] if pid == 2112]
        if nps:
            ach_leadn[i] = max(nps)
        w[i] = e["w"]
    return dict(p_pi=p_pi, p_N=p_N, pich=pich, Npid=Npid, pos0=pos0, npos=npos, nmom=nmom,
                nisp=nisp, nmask=nmask, w=w, ach_leadn=ach_leadn, A=A, n=n)


def ado_leadn(nterms):
    """ADoNIS per-event leading (max-|p|) ejected-neutron |p| (-1 if none)."""
    g0 = nterms[0]
    sp = np.asarray(g0["species"]); ch = np.asarray(g0["charge"]); al = np.asarray(g0["alive"])
    p3 = np.linalg.norm(np.asarray(g0["p4"])[:, :, 1:], axis=2)
    isn = (sp == NUCLEON) & (ch == 0) & al & (p3 > 0)
    return np.where(isn, p3, -1.0).max(1)


def wbin(vals, w, edges):
    m = vals >= 0
    h, _ = np.histogram(vals[m], edges, weights=w[m]); h2, _ = np.histogram(vals[m], edges, weights=w[m] ** 2)
    return h, np.sqrt(h2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dump"); ap.add_argument("target", nargs="?", default="C")
    ap.add_argument("--no-pauli", action="store_true"); ap.add_argument("--no-inel", action="store_true")
    a = ap.parse_args()
    print(f"parsing {a.dump} ...", flush=True)
    events = parse_cascadedump(a.dump)
    arr = res_arrays(events); n = arr["n"]
    print(f"  {n} RES events; bg A={arr['A']}  mean bg p/n={arr['nisp'].sum(1).mean():.2f}/"
          f"{(arr['nmask'].sum(1) - arr['nisp'].sum(1)).mean():.2f}  "
          f"pion charge frac +/0/-={np.bincount(arr['pich'], minlength=3) / n}", flush=True)
    cfg = build_cfg(a.target, pauli=not a.no_pauli, inel=not a.no_inel)
    key = jax.random.PRNGKey(12345)
    su = dict(npos=jnp.asarray(arr["npos"]), nmom=jnp.asarray(arr["nmom"]), nisp=jnp.asarray(arr["nisp"]),
              pos0=jnp.asarray(arr["pos0"]), consumed0=jnp.asarray(~arr["nmask"]),
              ch0=jnp.asarray(arr["pich"]), kp=key)
    p_pi = jnp.asarray(arr["p_pi"]); p_N = jnp.asarray(arr["p_N"])
    ppid = jnp.full(n, 211, jnp.int32)                  # unused (su_external bypasses setup_nucleus)
    ipid = jnp.full(n, 2112, jnp.int32)
    Npid = jnp.asarray(arr["Npid"]).astype(jnp.int32)
    print(f"running ADoNIS pool RES cascade on ACHILLES input (pauli={not a.no_pauli} inel={not a.no_inel}) ...", flush=True)
    out = cascade_nucleus(p_pi, p_N, ppid, ipid, Npid, cfg, key, P=4, channel="res", su_external=su)
    nterms = out[1]
    ado = ado_leadn(nterms); ach = arr["ach_leadn"]; w = arr["w"]; W = w.sum()
    da, ea = wbin(ado, w, EDGES); dh, eh = wbin(ach, w, EDGES)
    print("\n=== IDENTICAL INPUT: ADoNIS transport vs ACHILLES transport, LEADING-neutron |p| ===", flush=True)
    print(f"  events with >=1 final neutron: ADoNIS={ (ado>=0).sum()}  ACHILLES={ (ach>=0).sum()}", flush=True)
    print(f"  {'|p| bin':12s} {'ADO':>11s} {'ACH':>11s} {'ACH/ADO':>8s} {'pull':>7s}", flush=True)
    for k in range(len(EDGES) - 1):
        pull = (dh[k] - da[k]) / np.sqrt(ea[k] ** 2 + eh[k] ** 2 + 1e-300)
        print(f"  {EDGES[k]:5.0f}-{EDGES[k+1]:<5.0f}  {da[k]:.4e} {dh[k]:.4e} {dh[k]/max(da[k],1e-30):8.3f} {pull:+7.2f}", flush=True)
    # soft (<150) summary -- the disagreement corner
    soft_a = da[0] + da[1]; soft_h = dh[0] + dh[1]
    print(f"\n  soft (<150 MeV) leading-neutron: ADO={soft_a:.4e}  ACH={soft_h:.4e}  ACH/ADO={soft_h/max(soft_a,1e-30):.3f}", flush=True)
    tot_a = da.sum(); tot_h = dh.sum()
    print(f"  total leading-neutron:           ADO={tot_a:.4e}  ACH={tot_h:.4e}  ACH/ADO={tot_h/max(tot_a,1e-30):.3f}", flush=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from adonis.workflow.plotting import chi2_ratio_panel
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1], sharex=True)
    ma = ado >= 0; mh = ach >= 0
    chi2_ratio_panel(a0, a1, EDGES, {"values": ach[mh], "w": w[mh]}, {"values": ado[ma], "w": w[ma]},
                     label="ablation: leading-neutron |p| [MeV] (identical RES input)",
                     ado_label="ADoNIS", ref_label="ACHILLES", logy=True)
    a0.set_ylabel("dsigma/bin (paired input)"); a1.set_ylabel("ACH/ADO")
    a0.legend(fontsize=8); a0.set_title(f"RES cascade ablation (identical input): leading-neutron |p| "
                                        f"(pauli={not a.no_pauli} inel={not a.no_inel})", fontsize=9)
    tag = ("_nopauli" if a.no_pauli else "") + ("_noinel" if a.no_inel else "")
    o = f"/tmp/ablation_res_leadn{tag}.png"; fig.tight_layout(); fig.savefig(o, dpi=120)
    print(f"\n  wrote {o}", flush=True)


if __name__ == "__main__":
    main()
