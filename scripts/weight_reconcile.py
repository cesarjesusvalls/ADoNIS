"""Full per-event WEIGHT reconciliation on ACHILLES self-consistent RESDUMP events
(amps2, flux, initwgt, spinavg, psw all from the SAME CrossSection call). Free proton mono.
dsigma/dQ2 shape = bin(amps2 * flux * initwgt * spinavg * psw). flux/initwgt/spinavg are
constant, so only amps2 (proven correct) and psw (phase space) shape it. This checks each
factor mine-vs-ACHILLES and builds the self-consistent ACHILLES dsigma/dW,dQ2 oracle."""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.xsec.dcc_current import exclusive_amps2_batch
from adonis.xsec.backend import flux_factor, MASS_PDG_PROTON

LOG = Path("/Users/homelab/Lab/Playground/projects/DIFFGEN/Achilles/_zmtxout/lhdump.txt")


def vec(s): return np.array([float(x) for x in s.split(",")])


def parse():
    rows = []
    for line in open(LOG):
        if not line.startswith("RESDUMP"): continue
        d = dict(re.findall(r"(\w+)=([-+0-9.eE,]+)", line))
        if float(d["amps2"]) <= 0: continue
        knu = np.concatenate([[float(d["liE"])], vec(d["li"])])
        kmu = np.concatenate([[float(d["loE"])], vec(d["lo"])])
        pst = np.concatenate([[float(d["hiE"])], vec(d["hi"])])
        pN = np.concatenate([[float(d["hNE"])], vec(d["hN"])])
        ppi = np.concatenate([[float(d["hPE"])], vec(d["hP"])])
        rows.append((float(d["amps2"]), float(d["flux"]), float(d["initwgt"]),
                     float(d["spinavg"]), float(d["psw"]), knu, kmu, pst, pN, ppi))
    return rows


def main():
    rows = parse()
    print(f"parsed {len(rows)} self-consistent nonzero RESDUMP events")
    amps2 = np.array([r[0] for r in rows]); flux = np.array([r[1] for r in rows])
    initwgt = np.array([r[2] for r in rows]); spinavg = np.array([r[3] for r in rows])
    psw = np.array([r[4] for r in rows])
    knu = np.array([r[5] for r in rows]); kmu = np.array([r[6] for r in rows])
    pst = np.array([r[7] for r in rows]); pN = np.array([r[8] for r in rows]); ppi = np.array([r[9] for r in rows])
    q = knu - kmu; Q2 = ((q[:, 1:] ** 2).sum(1) - q[:, 0] ** 2) / 1e6
    pcm = pN + ppi; W = np.sqrt(np.clip(pcm[:, 0] ** 2 - (pcm[:, 1:] ** 2).sum(1), 0, None))

    # my amps2 + my flux on the same events
    my = np.zeros(len(rows)); ch = 500
    for i in range(0, len(rows), ch):
        sl = slice(i, min(i + ch, len(rows)))
        my[sl] = np.asarray(exclusive_amps2_batch(knu[sl], kmu[sl], pst[sl], pN[sl], ppi[sl], +1, 211))
    myflux = np.asarray(flux_factor(knu, pst, had_mass=MASS_PDG_PROTON))

    print(f"\n  factor reconciliation (mine/ACHILLES), free proton mono:")
    print(f"   amps2 : median {np.median(my[my>0]/amps2[my>0]):.4f}   (Q2-binned below)")
    print(f"   flux  : median {np.median(myflux/flux):.4f}   ACH flux={flux[0]:.4e}  mine={myflux[0]:.4e}")
    print(f"   initwgt: ACH all == {np.unique(initwgt)}   spinavg: ACH all == {np.unique(spinavg)}")
    ok = (my > 0) & np.isfinite(my)
    print(f"\n  amps2 mine/ACH vs Q^2:")
    bins = np.linspace(0, 1.2, 13)
    for b in range(len(bins) - 1):
        m = ok & (Q2 >= bins[b]) & (Q2 < bins[b + 1])
        if m.sum() > 3:
            print(f"   Q2={0.5*(bins[b]+bins[b+1]):.2f} n={m.sum():5d}  amps2 mine/ach={np.median(my[m]/amps2[m]):.3f}")

    # self-consistent ACHILLES dsigma/dW, dQ2 from RESDUMP weights = amps2*flux*initwgt*spinavg*psw
    wach = amps2 * flux * initwgt * spinavg * psw
    wmine = my * myflux * initwgt * spinavg * psw          # mine: same psw (ACH proposal), my amps2+flux
    def shp(v, w, e):
        h, _ = np.histogram(v, bins=e, weights=w); c = 0.5 * (e[1:] + e[:-1])
        a = np.sum(0.5 * (h[1:] + h[:-1]) * np.diff(c)); return c, (h / a if a else h)
    We = np.linspace(1080, 1500, 26); Qe = np.linspace(0, 1.2e6, 26)
    cW, hoW = shp(W, wach, We); _, hmW = shp(W, wmine, We)
    cQ, hoQ = shp(Q2 * 1e6, wach, Qe); _, hmQ = shp(Q2 * 1e6, wmine, Qe)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].step(cW, hoW, where="mid", color="0.4", label="ACHILLES (self-consistent)")
    ax[0].step(cW, hmW, where="mid", color="C3", label="mine (my amps2+flux, ACH psw)")
    ax[0].legend(); ax[0].set_xlabel("W [MeV]"); ax[0].set_title("dσ/dW shape (RESDUMP reweight)")
    ax[1].step(cQ / 1e6, hoQ, where="mid", color="0.4", label="ACHILLES")
    ax[1].step(cQ / 1e6, hmQ, where="mid", color="C3", label="mine")
    ax[1].legend(); ax[1].set_xlabel("Q² [GeV²]"); ax[1].set_title("dσ/dQ² shape")
    fig.tight_layout(); fig.savefig("figures/weight_reconcile.png", dpi=110)
    print("\n  wrote figures/weight_reconcile.png  (if mine==ACH here, amps2+flux+psw are all consistent;")
    print("   the ONLY thing left that my generator must independently reproduce is the psw = phase space)")


if __name__ == "__main__":
    main()
