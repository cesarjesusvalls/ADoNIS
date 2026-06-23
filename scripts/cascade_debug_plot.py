"""Plot the CC1pi pion-fate ADoNIS-vs-ACHILLES comparison per initial-|p_pi| bracket:
survived-pi+, charge-exchange, absorbed (fractions + binomial errors), with a pull row.
Run: python -u scripts/cascade_debug_plot.py [ado.npz] [ach.fate] [out.png]
"""
import sys, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ado_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ado_cascade_debug_C.npz"
ach_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ach_fatepion_C_gauss.fate"
out_png = sys.argv[3] if len(sys.argv) > 3 else "paper_figures/cc1pi_cascade_cex.png"
EDGES = np.array([0, 100, 150, 200, 250, 300, 350, 400, 500, 700, 1000.0])  # cap top bin for display
CTR = 0.5 * (EDGES[1:] + EDGES[:-1]); _PI = (211, 111, -211)

# ACHILLES: pair each event's primary-pion pin with its FATE_FS final-state multiplicity
rxF = re.compile(r"^FATE pidin=(-?\d+) pin=([\d.eE+-]+) .* status=(\d+)")
rxFS = re.compile(r"^FATE_FS np=(\d+) nn=(\d+) npip=(\d+) npi0=(\d+) npim=(\d+)")
apin, asurv, acex, aabs = [], [], [], []; cur = None
for ln in open(ach_path):
    m = rxF.match(ln)
    if m:
        if int(m[1]) in _PI: cur = float(m[2])
        continue
    m = rxFS.match(ln)
    if m and cur is not None:
        npip, npi0, npim = int(m[3]), int(m[4]), int(m[5])
        apin.append(cur); asurv.append(npip >= 1); acex.append((npip == 0) and (npi0 + npim >= 1))
        aabs.append(npip + npi0 + npim == 0); cur = None
apin = np.array(apin); A = {"surv": np.array(asurv), "cex": np.array(acex), "abs": np.array(aabs)}

# ADoNIS: classify identically from final-state multiplicity
d = dict(np.load(ado_path)); w = d["w"]; pin = d["p_init"]; pip = d["ch_init"] == 0
npip, npi0, npim = d["npip"], d["npi0"], d["npim"]
D = {"surv": npip >= 1, "cex": (npip == 0) & ((npi0 + npim) >= 1), "abs": (npip + npi0 + npim) == 0}
neff = lambda x: (x.sum() ** 2 / (x * x).sum()) if x.sum() > 0 else 0.0
be = lambda p, n: np.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))

def series(key):
    fD, eD, fA, eA, pull = [], [], [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        bD = pip & (pin >= lo) & (pin < hi); bA = (apin >= lo) & (apin < hi)
        wb = w[bD]; nD = neff(wb); fd = float((wb * D[key][bD]).sum() / wb.sum())
        nA = int(bA.sum()); fa = float(A[key][bA].mean())
        ed, ea = be(fd, nD), be(fa, nA); de = np.hypot(ed, ea)
        fD.append(fd); eD.append(ed); fA.append(fa); eA.append(ea); pull.append((fd - fa) / de if de > 0 else 0)
    return map(np.array, (fD, eD, fA, eA, pull))

fig, ax = plt.subplots(2, 3, figsize=(15, 7), height_ratios=[3, 1], sharex="col")
titles = [("surv", "survived $\\pi^+$ (CC1$\\pi$ signal pion)"), ("cex", "charge-exchange $\\pi^+\\to\\pi^0$"),
          ("abs", "absorbed")]
for c, (key, ttl) in enumerate(titles):
    fD, eD, fA, eA, pull = series(key)
    a0, a1 = ax[0, c], ax[1, c]
    a0.errorbar(CTR, fA, yerr=eA, fmt="s-", color="0.35", ms=5, capsize=2, lw=1.4, label="ACHILLES")
    a0.errorbar(CTR, fD, yerr=eD, fmt="o-", color="C3", ms=5, capsize=2, lw=1.4, label="ADoNIS (pool)")
    a0.set_title(ttl, fontsize=11); a0.set_ylim(0, 1); a0.grid(alpha=0.3)
    if c == 0: a0.set_ylabel("fraction of primary $\\pi^+$"); a0.legend(fontsize=9)
    chi2 = float(np.mean(pull ** 2)); a0.text(0.04, 0.92, f"$\\chi^2$/ndf={chi2:.0f}", transform=a0.transAxes,
              fontsize=9, va="top")
    a1.axhspan(-2, 2, color="green", alpha=0.12); a1.axhline(0, color="0.5", lw=0.8)
    a1.bar(CTR, pull, width=np.diff(EDGES) * 0.8, color=["C3" if abs(p) >= 3 else "0.6" for p in pull])
    a1.set_xlabel("initial $|p_\\pi|$ [MeV/c]"); a1.set_ylim(-70, 70)
    if c == 0: a1.set_ylabel("pull ($\\sigma$)")
fig.suptitle("Pion fate in the carbon cascade, by initial $|p_\\pi|$: ADoNIS vs ACHILLES (Gaussian, ~550k $\\pi^+$/side)\n"
             "absorption agrees; ADoNIS charge-exchange is systematically too low (excess shows as surviving $\\pi^+$)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out_png, dpi=130)
print("wrote", out_png)
