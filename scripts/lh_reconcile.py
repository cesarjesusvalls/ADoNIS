"""DECISIVE reconcile: ACHILLES dumps (lepton L, hadron H, amps2) and the SAME-CALL momenta
from XSecBackend (self-consistent, unlike the RESGEN dump). Compute my exclusive_amps2_batch
on those exact momenta and bin my/ach vs Q^2. If flat ~1.0, the generator is correct and the
resgen-bench 0.5x was a stale-g_last_amps2 artifact."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
from adonis.xsec.dcc_current import exclusive_amps2_batch

LOG = Path("/Users/homelab/Lab/Playground/projects/DIFFGEN/Achilles/_zmtxout/lhdump.txt")


def parse():
    a, knu, kmu, pN, ppi, pst = [], [], [], [], [], []
    cur = None
    for line in open(LOG):
        if line.startswith("LHDUMP"):
            cur = {"amps2": float(line.split("amps2=")[1].split()[0]), "hadout": []}
        elif cur is None:
            continue
        elif line.startswith("LH_lepin"):  cur["knu"] = [float(x) for x in line.split()[1:5]]
        elif line.startswith("LH_lepout"): cur["kmu"] = [float(x) for x in line.split()[1:5]]
        elif line.startswith("LH_hadin"):  cur["hadin"] = [float(x) for x in line.split()[1:5]]
        elif line.startswith("LH_hadout"): cur["hadout"].append([float(x) for x in line.split()[1:5]])
        elif line.startswith("LH_H"):      # only nonzero-hadron events reach here
            if cur.get("amps2", 0) > 0 and len(cur["hadout"]) == 2 and not cur.get("_done"):
                cur["_done"] = True
                a.append(cur["amps2"]); knu.append(cur["knu"]); kmu.append(cur["kmu"])
                pst.append(cur["hadin"]); pN.append(cur["hadout"][0]); ppi.append(cur["hadout"][1])
    return (np.array(a), np.array(knu), np.array(kmu), np.array(pst), np.array(pN), np.array(ppi))


def main():
    a, knu, kmu, pst, pN, ppi = parse()
    print(f"parsed {len(a)} self-consistent nonzero events", flush=True)
    q = knu - kmu
    Q2 = ((q[:, 1:] ** 2).sum(1) - q[:, 0] ** 2) / 1e6
    pcm = pN + ppi
    W = np.sqrt(np.clip(pcm[:, 0] ** 2 - (pcm[:, 1:] ** 2).sum(1), 0, None))
    # my amps2 in chunks
    n = len(a); my = np.zeros(n); ch = 500
    for i in range(0, n, ch):
        sl = slice(i, min(i + ch, n))
        my[sl] = np.asarray(exclusive_amps2_batch(knu[sl], kmu[sl], pst[sl], pN[sl], ppi[sl], +1, 211))
    ok = np.isfinite(my) & (my > 0) & (a > 0)
    r = my[ok] / a[ok]
    print(f"\noverall: n_ok={ok.sum()}/{n}  global sum_my/sum_ach={my[ok].sum()/a[ok].sum():.4f}")
    print("\n  my(exclusive_amps2_batch) / ACHILLES amps2,  SELF-CONSISTENT kinematics, vs Q^2:")
    bins = np.linspace(0, 1.2, 13)
    Q2o = Q2[ok]
    for b in range(len(bins) - 1):
        m = (Q2o >= bins[b]) & (Q2o < bins[b + 1])
        if m.sum() > 3:
            print(f"   Q2={0.5*(bins[b]+bins[b+1]):.2f}  n={m.sum():5d}  "
                  f"<my/ach>={np.mean(r[m]):.3f}  median={np.median(r[m]):.3f}  "
                  f"sumratio={my[ok][m].sum()/a[ok][m].sum():.3f}")


if __name__ == "__main__":
    main()
