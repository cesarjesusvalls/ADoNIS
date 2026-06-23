"""CC1pi cascade-tension comparison: ADoNIS full-pool dump (cascade_debug_dump.py) vs ACHILLES
fatepion2 FATE/FATE_FS, per INITIAL-pion-|p| bracket, in CHI2/PULL terms (significance, not ratio).

ACHILLES parsing: the dump emits, per primary cascade particle, a `FATE pidin=.. pin=.. status=..` line
(status 1=transmitted/never-interacted, 29=reacted), and per EVENT a `FATE_FS np nn npip npi0 npim`
(final-state multiplicity).  We group lines into events (a FATE_FS closes an event), take the primary
PION's pin (pidin in +/-211,111) as the initial |p|, and read the event's final pion content from
FATE_FS:  survived-pi+ = npip>=1 ; absorbed = npip+npi0+npim==0 ; charge-ex = npip==0 & (npi0+npim)>=1.
RES-only: events with a primary-pion FATE line (QE events, no primary pion, are skipped).

Each bracket fraction f gets a binomial error sqrt(f(1-f)/N) -- ADoNIS N = effective (Sum w)^2/Sum w^2,
ACHILLES N = raw count.  pull = (f_ado - f_ach)/sqrt(e_ado^2 + e_ach^2); chi2/ndf over brackets; >=3sigma
brackets flagged (*).  This is the deviation-trend table that drives the code-reading.

Run: python -u scripts/cascade_debug_compare.py <ado_dump.npz> <ach.fate>
"""
import sys, re
import numpy as np
np.seterr(all="ignore")

ado_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ado_cascade_debug_C.npz"
ach_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ach_fatepion_C_gauss.fate"
PI_EDGES = np.array([0, 100, 150, 200, 250, 300, 350, 400, 500, 700, 2000.0])
_PIPIDS = (211, 111, -211)

# ---------------- ACHILLES: group FATE lines into events, pair pion-pin with FATE_FS ----------------
rxF = re.compile(r"^FATE pidin=(-?\d+) pin=([\d.eE+-]+) pidout=(-?\d+) pout=([\d.eE+-]+) status=(\d+)")
rxFS = re.compile(r"^FATE_FS np=(\d+) nn=(\d+) npip=(\d+) npi0=(\d+) npim=(\d+)")
ach_pin, ach_surv, ach_chgex, ach_abs, ach_np = [], [], [], [], []
cur_pion_pin = None
for ln in open(ach_path):
    mF = rxF.match(ln)
    if mF:
        # APPLES-TO-APPLES: bracket ONLY by a pi+ PRIMARY (pidin==211), matching the ADoNIS pi+ dump.
        # RES also makes pi0/pi- primaries (which charge-exchange readily); folding them in against an
        # ADoNIS pi+-only sample manufactured a spurious cex deficit -- the original (wrong) trend.
        if int(mF[1]) == 211:
            cur_pion_pin = float(mF[2])
        continue
    mFS = rxFS.match(ln)
    if mFS:
        if cur_pion_pin is not None:             # RES event with a pi+ primary
            npip, npi0, npim = int(mFS[3]), int(mFS[4]), int(mFS[5])
            ach_pin.append(cur_pion_pin)
            ach_surv.append(npip >= 1)
            ach_abs.append(npip + npi0 + npim == 0)
            ach_chgex.append((npip == 0) and (npi0 + npim >= 1))
            ach_np.append(int(mFS[1]))
        cur_pion_pin = None                      # close the event
ach_pin = np.array(ach_pin); ach_surv = np.array(ach_surv); ach_chgex = np.array(ach_chgex)
ach_abs = np.array(ach_abs); ach_np = np.array(ach_np, float)
print(f"ACHILLES RES pion events parsed: {len(ach_pin)}", flush=True)

# ---------------- ADoNIS dump ----------------
d = dict(np.load(ado_path))
w = d["w"]; pin = d["p_init"]; ch0 = d["ch_init"]
# Classify BOTH sides identically, from the EVENT-LEVEL final-state pion multiplicity (npip/npi0/npim) --
# the same quantity ACHILLES FATE_FS reports -- so the comparison is apples-to-apples (a primary absorbed
# but a cascade-created pion is treated the same way on both sides).
npip, npi0, npim = d["npip"], d["npi0"], d["npim"]
ado_surv = npip >= 1                             # >=1 final pi+ (CC1pi signal-relevant)
ado_abs = (npip + npi0 + npim) == 0              # no final pion (absorbed)
ado_chgex = (npip == 0) & ((npi0 + npim) >= 1)   # final pion(s) but no pi+ (charge-exchange)
pip = ch0 == 0                                   # pi+ primaries (the CC1pi signal pion)
print(f"ADoNIS RES pi+ events: {int(pip.sum())}  (overflow stack={int(d['overflow_stack'])} out={int(d['overflow_out'])}, "
      f"max_steps={int(d['max_steps'])})", flush=True)


def neff(x):
    s = x.sum(); return float(s * s / (x * x).sum()) if s > 0 else 0.0


def be(p, n):
    return float(np.sqrt(max(p * (1 - p), 1e-9) / max(n, 1)))


def compare(label, ado_mask, ach_mask):
    print(f"\n=== {label}  ADoNIS vs ACHILLES per initial-|p_pi| bracket (pi+) ===")
    print(f"{'p_in bin':>13s} {'N_ach':>8s} {'ADoNIS':>16s} {'ACHILLES':>16s} {'pull(sigma)':>12s}")
    chi2 = 0.0; ndf = 0
    for lo, hi in zip(PI_EDGES[:-1], PI_EDGES[1:]):
        bD = pip & (pin >= lo) & (pin < hi)
        bA = (ach_pin >= lo) & (ach_pin < hi)
        if not bD.any() or bA.sum() == 0:
            continue
        wb = w[bD]; nD = neff(wb)
        fD = float((wb * ado_mask[bD]).sum() / wb.sum()); eD = be(fD, nD)
        nA = int(bA.sum()); fA = float(ach_mask[bA].mean()); eA = be(fA, nA)
        de = np.sqrt(eD * eD + eA * eA); pull = (fD - fA) / de if de > 0 else 0.0
        chi2 += pull * pull; ndf += 1
        flag = "  ***" if abs(pull) >= 3 else ("  *" if abs(pull) >= 2 else "")
        print(f"{f'[{lo:.0f},{hi:.0f})':>13s} {nA:8d} {fD:7.4f}±{eD:.4f} {fA:7.4f}±{eA:.4f} {pull:+8.2f}{flag}")
    print(f"  chi2/ndf = {chi2/max(ndf,1):.2f}  (ndf={ndf})")
    return chi2, ndf


tot = 0.0; tn = 0
for lbl, am, hm in (("survived pi+", ado_surv, ach_surv),
                    ("charge-exchange", ado_chgex, ach_chgex),
                    ("absorbed", ado_abs, ach_abs)):
    c, n = compare(lbl, am, hm); tot += c; tn += n

# proton multiplicity (final-state np), pi+ bracket -- secondary (count only, no momentum cut on ACH side)
print(f"\n=== <final-state protons np> per initial-|p_pi| bracket (pi+)  [multiplicity, count] ===")
print(f"{'p_in bin':>13s} {'ADoNIS<np_esc>':>15s} {'ACHILLES<np>':>13s}")
for lo, hi in zip(PI_EDGES[:-1], PI_EDGES[1:]):
    bD = pip & (pin >= lo) & (pin < hi); bA = (ach_pin >= lo) & (ach_pin < hi)
    if not bD.any() or bA.sum() == 0: continue
    wb = w[bD]
    print(f"{f'[{lo:.0f},{hi:.0f})':>13s} {(wb*d['n_p_esc'][bD]).sum()/wb.sum():15.3f} {ach_np[bA].mean():13.3f}")

print(f"\nOVERALL pion-fate chi2/ndf = {tot/max(tn,1):.2f}  (ndf={tn}); brackets with |pull|>=3 flagged *** above")
