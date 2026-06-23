"""Show that the ADoNIS survived-pi+ EXCESS equals the charge-exchange DEFICIT vs ACHILLES, per
initial-|p_pi| bracket -- and that absorbed is unchanged (the non-trivial part; since surv+cex+abs=1
on both sides, dSurv+dCEX+dAbs==0 identically, so dAbs~=0 forces dSurv = -dCEX).
Run: python -u scripts/cascade_debug_complementarity.py
"""
import re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

EDGES = np.array([0, 100, 150, 200, 250, 300, 350, 400, 500, 700, 1000.])
CTR = 0.5 * (EDGES[1:] + EDGES[:-1]); _PI = (211, 111, -211)
rxF = re.compile(r"^FATE pidin=(-?\d+) pin=([\d.eE+-]+) .* status=(\d+)")
rxFS = re.compile(r"^FATE_FS np=(\d+) nn=(\d+) npip=(\d+) npi0=(\d+) npim=(\d+)")
apin = []; As = []; Ac = []; Aa = []; cur = None
for ln in open("/tmp/ach_fatepion_C_gauss.fate"):
    m = rxF.match(ln)
    if m:
        if int(m[1]) in _PI: cur = float(m[2])
        continue
    m = rxFS.match(ln)
    if m and cur is not None:
        pp, p0, pm = int(m[3]), int(m[4]), int(m[5])
        apin.append(cur); As.append(pp >= 1); Ac.append(pp == 0 and p0 + pm >= 1); Aa.append(pp + p0 + pm == 0); cur = None
apin = np.array(apin); A = dict(surv=np.array(As), cex=np.array(Ac), abs=np.array(Aa))
d = dict(np.load("/tmp/ado_cascade_debug_C.npz")); w = d["w"]; pin = d["p_init"]; pip = d["ch_init"] == 0
npip, npi0, npim = d["npip"], d["npi0"], d["npim"]
D = dict(surv=npip >= 1, cex=(npip == 0) & ((npi0 + npim) >= 1), abs=(npip + npi0 + npim) == 0)
neff = lambda x: (x.sum() ** 2 / (x * x).sum()) if x.sum() > 0 else 0.0
be = lambda p, n: np.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))

def fr(key):
    a = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        bD = pip & (pin >= lo) & (pin < hi); bA = (apin >= lo) & (apin < hi); wb = w[bD]
        fd = float((wb * D[key][bD]).sum() / wb.sum()); fa = float(A[key][bA].mean())
        a.append((fd - fa, np.hypot(be(fd, neff(wb)), be(fa, int(bA.sum())))))
    return np.array([x[0] for x in a]), np.array([x[1] for x in a])

dS, _ = fr("surv"); dC, eC = fr("cex"); dA, eA = fr("abs")
print(f"{'p_in':>11} {'dSurv':>8} {'-dCEX':>8} {'dAbs':>8} {'dAbs/err':>9}")
for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
    print(f"{f'[{lo:.0f},{hi:.0f})':>11} {dS[i]:+8.3f} {-dC[i]:+8.3f} {dA[i]:+8.3f} {dA[i]/eA[i]:+9.1f}")
print(f"\ncorr(dSurv,-dCEX)={np.corrcoef(dS,-dC)[0,1]:.4f}  max|dAbs|={np.abs(dA).max():.3f}  <|dAbs|/err>={np.mean(np.abs(dA)/eA):.2f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].axhline(0, color='0.7', lw=.8)
ax[0].plot(CTR, dS, 'o-', color='C0', label='$\\Delta$ survived $\\pi^+$')
ax[0].plot(CTR, -dC, 's--', color='C3', label='$-\\Delta$ charge-exchange')
ax[0].errorbar(CTR, dA, yerr=eA, fmt='^:', color='0.4', capsize=2, label='$\\Delta$ absorbed ($\\approx$0)')
ax[0].set_xlabel('initial $|p_\\pi|$ [MeV/c]'); ax[0].set_ylabel('ADoNIS $-$ ACHILLES (fraction)')
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3); ax[0].set_title('survived excess tracks cex deficit; absorbed $\\approx$0')
lim = max(dS.max(), (-dC).max()) * 1.1
ax[1].plot([0, lim], [0, lim], 'k--', lw=1, label='y=x')
sc = ax[1].scatter(-dC, dS, c=CTR, cmap='viridis', s=60, zorder=3)
ax[1].set_xlabel('$-\\Delta$ charge-exchange'); ax[1].set_ylabel('$\\Delta$ survived $\\pi^+$')
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3); ax[1].set_title(f'1:1  (corr={np.corrcoef(dS,-dC)[0,1]:.3f})')
cb = fig.colorbar(sc, ax=ax[1]); cb.set_label('$|p_\\pi|$ [MeV/c]')
fig.tight_layout(); fig.savefig("paper_figures/cc1pi_cex_complementarity.png", dpi=130)
print("wrote paper_figures/cc1pi_cex_complementarity.png")
