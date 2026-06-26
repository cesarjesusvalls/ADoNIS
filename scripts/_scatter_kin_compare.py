"""Isolate the NN-elastic 2->2 recoil kinematics: take ACHILLES's exact per-scatter inputs (CDSCAT
in1=leading, in2=struck) for pn scatters (struck neutron) and run ADoNIS's _two_body_cm_scatter on the
SAME inputs, then compare the recoil-neutron |p| distribution to ACHILLES's own recoil (CDSCAT out).
Same inputs + (both isotropic) => identical distribution IF the 2->2 kinematics match.  A narrower ADoNIS
recoil => a frame/boost/sampling difference in the 2->2 (not Pauli/capture/sub-cascade).

Run: NMAX=300000 python -u scripts/_scatter_kin_compare.py _oracle_out/QE_cascdump.cdump
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from adonis.fsi.cascade_real import _two_body_cm_scatter
from adonis.constants import mp as MP, mn as MN

DUMP = sys.argv[1] if len(sys.argv) > 1 else "_oracle_out/QE_cascdump.cdump"
NMAX = int(os.environ.get("NMAX", "300000"))


def vec(s): return [float(x) for x in s.split(",")]


def main():
    in1 = []; in2 = []; ach_rec = []
    nsc = 0
    with open(DUMP) as f:
        for line in f:
            if not line.startswith("CDSCAT"):
                continue
            d = {}; outs = []
            for t in line.split()[1:]:
                if t.startswith("out="):
                    p = t[4:].split(","); outs.append((int(p[0]), [float(x) for x in p[1:]]))
                elif "=" in t:
                    k, v = t.split("=", 1); d[k] = v
            if int(d["type"]) != 1 or int(d["pid2"]) != 2112:     # elastic NN, struck = neutron
                continue
            rec = [o for o in outs if o[0] == 2112]               # recoil neutron = out 2112
            if not rec:
                continue
            in1.append(vec(d["in1"])); in2.append(vec(d["in2"])); ach_rec.append(rec[0][1])
            nsc += 1
            if nsc >= NMAX:
                break
    in1 = np.array(in1); in2 = np.array(in2); ach_rec = np.array(ach_rec)
    ach_p = np.linalg.norm(ach_rec[:, 1:], axis=1)
    n = len(in1); keys = jax.random.split(jax.random.PRNGKey(0), n)

    def scat(l, s, k):
        po = _two_body_cm_scatter(l, s, MP, k, m_recoil=MN)        # leading stays proton; recoil neutron
        r = (l + s) - po
        return jnp.linalg.norm(r[1:])
    ado_p = np.array(jax.vmap(scat)(jnp.array(in1), jnp.array(in2), keys))

    print(f"pn scatters compared: {n}", flush=True)
    edges = np.array([0, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 1000, 1500.])
    ha, _ = np.histogram(ado_p, edges); hh, _ = np.histogram(ach_p, edges)
    print(f"  {'recoil-n |p|':14s} {'ADO':>8s} {'ACH':>8s} {'ACH/ADO':>8s}", flush=True)
    for i in range(len(edges) - 1):
        print(f"  {edges[i]:5.0f}-{edges[i+1]:<5.0f} {ha[i]:>8d} {hh[i]:>8d} {hh[i]/max(ha[i],1):>8.3f}", flush=True)
    print(f"  median |p|: ADO={np.median(ado_p):.0f}  ACH={np.median(ach_p):.0f} MeV", flush=True)
    print(f"  frac 150-250: ADO={((ado_p>150)&(ado_p<250)).mean():.3f}  ACH={((ach_p>150)&(ach_p<250)).mean():.3f}", flush=True)
    print(f"  frac >700:    ADO={(ado_p>700).mean():.4f}  ACH={(ach_p>700).mean():.4f}", flush=True)
    print(f"  mean |p|: ADO={ado_p.mean():.1f}  ACH={ach_p.mean():.1f}  std ADO={ado_p.std():.1f} ACH={ach_p.std():.1f}", flush=True)


if __name__ == "__main__":
    main()
