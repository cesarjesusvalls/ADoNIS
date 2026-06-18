"""Parse ACHILLES ACHILLES_FATEDUMP stderr (FATE / FATE_FS lines) and compare the single-pass
primary-nucleon fates to the ADoNIS dump (scripts/cascade_fate_dump.py npz).

ACHILLES FATE line:  FATE pidin=<> pin=<MeV> pidout=<> pout=<MeV> status=<1=final_state,26=captured>
                     pout==pin (within eps) => unscattered (transparent).
Run: python -u scripts/cascade_fate_compare.py <achilles_stderr.log> <ado_fate_*.npz>
"""
import sys, re, numpy as np
np.seterr(all="ignore")

ach_log = sys.argv[1]
ado_npz = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ado_fate_Ar_proton.npz"

# ---- ACHILLES side ----
pin, pout, pidin, pidout, status = [], [], [], [], []
rx = re.compile(r"FATE pidin=(-?\d+) pin=([\d.eE+-]+) pidout=(-?\d+) pout=([\d.eE+-]+) status=(\d+)")
with open(ach_log) as f:
    for ln in f:
        m = rx.search(ln)
        if m:
            pidin.append(int(m.group(1))); pin.append(float(m.group(2)))
            pidout.append(int(m.group(3))); pout.append(float(m.group(4))); status.append(int(m.group(5)))
pin = np.array(pin); pout = np.array(pout); pidin = np.array(pidin); pidout = np.array(pidout); status = np.array(status)
print(f"ACHILLES primaries parsed: {len(pin)}")

def be(p, n):                                   # binomial std error
    return float(np.sqrt(max(p * (1 - p), 0) / max(n, 1)))

def classify_ach(species_pid):
    # ACHILLES marks a scattered primary `interacted`(29); a never-interacting one ends final_state(1);
    # bound -> captured(26).  (verified: status==1 => |pout-pin|~0, i.e. truly unscattered.)
    sel = pidin == species_pid
    n = int(sel.sum())
    if n == 0:
        return None
    st = status[sel]
    esc = float((st == 1).mean()); scat = float((st == 29).mean()); cap = float((st == 26).mean())
    other = float((~np.isin(st, [1, 29, 26])).mean())
    return dict(N=n, ESCAPED_FREE=(esc, be(esc, n)), SCATTERED=(scat, be(scat, n)),
                CAPTURED=(cap, be(cap, n)), OTHER=other)

# ---- ADoNIS side (fate codes: 0 ESCAPED_FREE, 1 ELASTIC, 2 INELASTIC, 3 RECAPTURED) ----
d = dict(np.load(ado_npz))
w = d["w"]; fate = d["fate"]; pinit = d["p_init"]
def neff(wm): return (wm.sum() ** 2) / (wm ** 2).sum() if wm.sum() > 0 else 0.0
def wf(mask, ws=None):                            # weighted fraction + Kish-neff std error
    ws = w if ws is None else ws
    p = float((ws * mask).sum() / ws.sum()); ne = neff(ws)
    return p, float(np.sqrt(max(p * (1 - p), 0) / max(ne, 1)))
ado_esc = wf(fate == 0); ado_scat = wf((fate == 1) | (fate == 2)); ado_cap = wf(fate == 3)
species = str(d["species"]); spid = 2212 if species == "proton" else 2112
a = classify_ach(spid)

print(f"\n=== single-pass primary fate: ADoNIS ({species}) vs ACHILLES  [value ± stat] ===")
print(f"  (ADoNIS N_eff~{neff(w):.0f} weighted; ACHILLES N={a['N']} unweighted)")
print(f"{'fate':16s} {'ADoNIS':>18s} {'ACHILLES':>18s} {'Δ(ADO-ACH)':>14s}")
for k, av in [("ESCAPED_FREE", ado_esc), ("SCATTERED", ado_scat), ("CAPTURED", ado_cap)]:
    hv = a[k]
    dv = av[0] - hv[0]; de = np.sqrt(av[1] ** 2 + hv[1] ** 2)
    print(f"{k:16s} {av[0]:8.4f} ± {av[1]:.4f} {hv[0]:8.4f} ± {hv[1]:.4f} {dv:+8.4f}±{de:.4f}")

# per-|p| bracket: ESCAPED_FREE / SCATTERED / CAPTURED for both, with uncertainties (weight-free response)
edges = [0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 5000]
ado_codes = {"ESC": (fate == 0), "SCAT": (fate == 1) | (fate == 2), "CAP": (fate == 3)}
ach_codes = {"ESC": (status == 1), "SCAT": (status == 29), "CAP": (status == 26)}
for ch in ["ESC", "SCAT", "CAP"]:
    print(f"\n=== {ch}  vs initial |p|   [ADoNIS | ACHILLES | Δ(ADO-ACH)] ===")
    print(f"{'p_in bin':>13s} {'N_ach':>7s} {'ADoNIS':>16s} {'ACHILLES':>16s} {'Δ':>15s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        ab = (pinit >= lo) & (pinit < hi)
        sel = (pidin == spid) & (pin >= lo) & (pin < hi)
        av = wf(ado_codes[ch][ab], w[ab]) if ab.any() else (float("nan"), 0.0)
        nA = int(sel.sum()); hp = float(ach_codes[ch][sel].mean()) if nA else float("nan")
        he = be(hp, nA)
        dv = av[0] - hp; de = np.sqrt(av[1] ** 2 + he ** 2)
        sig = abs(dv) / de if de > 0 else 0
        flag = " *" if sig >= 3 else ""
        print(f"{f'[{lo},{hi})':>13s} {nA:7d} {av[0]:7.4f} ± {av[1]:.4f} {hp:7.4f} ± {he:.4f} {dv:+7.4f}±{de:.4f}{flag}")
