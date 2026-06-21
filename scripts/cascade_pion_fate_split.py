"""ABSORBED / SCATTERED / CHARGE-EX split of primary pions, ADoNIS vs ACHILLES, per initial-|p| bin.

ACHILLES side: per-event pairing of the achilles:fatepion2 dump -- each event block is the primary
FATE lines (status 1=transmitted, 29=reacted) terminated by a FATE_FS line carrying the final-state
pion counts (npip/npi0/npim).  For events with EXACTLY ONE primary pion we classify it:
  status 1                         -> TRANSMITTED
  status 29, 0 final pions         -> ABSORBED
  status 29, >=1 final pi SAME chg -> SCATTERED (quasi-elastic)
  status 29, final pi DIFF chg     -> CHARGE_EX
ADoNIS side: scripts/cascade_pion_fate_dump.py npz fate codes (0 T,1 QE,2 CEX,3 ABS,4 CONV).

Run: python -u scripts/cascade_pion_fate_split.py <ach.fate> <ado_pionfate.npz> [211]
"""
import sys, re, numpy as np
np.seterr(all="ignore")
ach = sys.argv[1]; ado = sys.argv[2]; spid = int(sys.argv[3]) if len(sys.argv) > 3 else 211
chmap = {211: "pip", 111: "pi0", -211: "pim"}

rxF = re.compile(r"FATE pidin=(-?\d+) pin=([\d.eE+-]+) pidout=(-?\d+) pout=([\d.eE+-]+) status=(\d+)")
rxFS = re.compile(r"FATE_FS np=(\d+) nn=(\d+) npip=(\d+) npi0=(\d+) npim=(\d+)")

# ACHILLES: walk event blocks (FATE... then FATE_FS), classify single-primary-pion events
rows = []  # (pin, charge_pid, fate)  fate: 0 T,1 QE,2 CEX,3 ABS
cur = []
for ln in open(ach):
    mF = rxF.search(ln)
    if mF:
        cur.append((int(mF[1]), float(mF[2]), int(mF[5]))); continue
    mS = rxFS.search(ln)
    if mS:
        npip, npi0, npim = int(mS[3]), int(mS[4]), int(mS[5])
        pis = [(pid, pin, st) for (pid, pin, st) in cur if pid in chmap]
        if len(pis) == 1:                                    # clean single-primary-pion event
            pid, pin, st = pis[0]
            nf = {211: npip, 111: npi0, -211: npim}
            if st == 1:
                fate = 0
            else:                                            # reacted
                if npip + npi0 + npim == 0: fate = 3         # absorbed
                elif nf[pid] >= 1:          fate = 1         # survived same charge
                else:                       fate = 2         # charge-exchange
            rows.append((pid, pin, fate))
        cur = []
A = np.array(rows, float) if rows else np.zeros((0, 3))
mA = A[:, 0] == spid
pinA = A[mA, 1]; fA = A[mA, 2].astype(int)
print(f"ACHILLES single-pion events pid={spid}: {int(mA.sum())}  "
      f"(T={int((fA==0).sum())} QE={int((fA==1).sum())} CEX={int((fA==2).sum())} ABS={int((fA==3).sum())})")

# ADoNIS
d = dict(np.load(ado)); w = d["w"]; fate = d["fate"]; pinit = d["p_init"]; ch = d["ch_init"]
chw = {211: 0, 111: 1, -211: 2}[spid]; mD = ch == chw
wD, fD, pD = w[mD], fate[mD], pinit[mD]
def neff(x): return (x.sum()**2)/(x**2).sum() if x.sum() > 0 else 0.0
def be(p, n): return float(np.sqrt(max(p*(1-p), 0)/max(n, 1)))
names = {0: "TRANSMIT", 1: "SCATTER", 2: "CHARGE_EX", 3: "ABSORBED"}
edges = [0, 100, 150, 200, 250, 300, 350, 400, 500, 700, 2000]
for code in (3, 1, 2, 0):
    print(f"\n=== {names[code]} fraction vs |p|   [ADoNIS | ACHILLES | Δ]  (* = >=3σ) ===")
    print(f"{'p_in bin':>13s} {'N_ach':>7s} {'ADoNIS':>16s} {'ACHILLES':>16s} {'Δ':>15s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        ab = (pD >= lo) & (pD < hi); sA = mA & (A[:, 1] >= lo) & (A[:, 1] < hi)
        nA = int(sA.sum())
        if not ab.any() or nA == 0: continue
        wb = wD[ab]; ne = neff(wb)
        pa = float((wb*(fD[ab] == code)).sum()/wb.sum()); ea = be(pa, ne)
        fAb = A[sA, 2].astype(int); ph = float((fAb == code).mean()); eh = be(ph, nA)
        dv = pa - ph; de = np.sqrt(ea**2 + eh**2); sig = abs(dv)/de if de > 0 else 0
        print(f"{f'[{lo},{hi})':>13s} {nA:7d} {pa:7.4f} ± {ea:.4f} {ph:7.4f} ± {eh:.4f} {dv:+7.4f}±{de:.4f}{' *' if sig>=3 else ''}")
