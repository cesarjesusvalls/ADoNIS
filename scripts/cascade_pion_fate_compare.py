"""Compare ADoNIS primary-PION single-pass fate (scripts/cascade_pion_fate_dump.py npz) to ACHILLES
achilles:fatepion FATE lines, per initial-|p| bin -- the pion analog of cascade_fate_compare.py.

ACHILLES FATE line (per primary, pid-agnostic): status 1 = never interacted (TRANSMITTED),
29 = interacted (REACTED).  The dump snapshots the INITIAL pion (pout==pin), so absorbed vs scattered
is NOT separable from these lines alone -- this tool does the TRANSMITTED vs REACTED (transparency)
comparison.  (Absorbed/scattered split needs the FATE_FS-pion extension.)

Run: python -u scripts/cascade_pion_fate_compare.py <ach_fate.fate> <ado_pionfate.npz> [211|111]
"""
import sys, re, numpy as np
np.seterr(all="ignore")
ach = sys.argv[1]; ado = sys.argv[2]; spid = int(sys.argv[3]) if len(sys.argv) > 3 else 211

rx = re.compile(r"FATE pidin=(-?\d+) pin=([\d.eE+-]+) pidout=(-?\d+) pout=([\d.eE+-]+) status=(\d+)")
P = {"pidin": [], "pin": [], "status": []}
for ln in open(ach):
    m = rx.search(ln)
    if m:
        P["pidin"].append(int(m[1])); P["pin"].append(float(m[2])); P["status"].append(int(m[5]))
pidin = np.array(P["pidin"]); pin = np.array(P["pin"]); status = np.array(P["status"])
mA = pidin == spid
print(f"ACHILLES primary pions pid={spid}: {int(mA.sum())}  (status1={int((status[mA]==1).sum())} status29={int((status[mA]==29).sum())})")

d = dict(np.load(ado))
w = d["w"]; fate = d["fate"]; pinit = d["p_init"]; ch_init = d["ch_init"]
ch_want = {211: 0, 111: 1, -211: 2}[spid]
mD = ch_init == ch_want
wD = w[mD]; fateD = fate[mD]; pD = pinit[mD]
def neff(x): return (x.sum()**2)/(x**2).sum() if x.sum() > 0 else 0.0
def be(p, n): return float(np.sqrt(max(p*(1-p), 0)/max(n, 1)))

print(f"\n=== TRANSMITTED vs REACTED  primary pi (pid={spid})  ADoNIS vs ACHILLES  [value ± stat] ===")
edges = [0, 100, 150, 200, 250, 300, 350, 400, 500, 700, 2000]
print(f"{'p_in bin':>13s} {'N_ach':>7s} {'ADoNIS_react':>16s} {'ACH_react':>16s} {'Δ(ADO-ACH)':>15s}")
for lo, hi in zip(edges[:-1], edges[1:]):
    ab = (pD >= lo) & (pD < hi); sA = mA & (pin >= lo) & (pin < hi)
    if not ab.any() or sA.sum() == 0: continue
    wb = wD[ab]; ne = neff(wb)
    pr_ado = float((wb*(fateD[ab] > 0)).sum()/wb.sum()); e_ado = be(pr_ado, ne)
    nA = int(sA.sum()); pr_ach = float((status[sA] == 29).mean()); e_ach = be(pr_ach, nA)
    dv = pr_ado - pr_ach; de = np.sqrt(e_ado**2 + e_ach**2); sig = abs(dv)/de if de > 0 else 0
    flag = " *" if sig >= 3 else ""
    print(f"{f'[{lo},{hi})':>13s} {nA:7d} {pr_ado:7.4f} ± {e_ado:.4f} {pr_ach:7.4f} ± {e_ach:.4f} {dv:+7.4f}±{de:.4f}{flag}")
# overall
ne = neff(wD); pr_ado = float((wD*(fateD > 0)).sum()/wD.sum())
nA = int(mA.sum()); pr_ach = float((status[mA] == 29).mean())
print(f"\nOVERALL reacted: ADoNIS {pr_ado:.4f}±{be(pr_ado,ne):.4f}  ACHILLES {pr_ach:.4f}±{be(pr_ach,nA):.4f}")
