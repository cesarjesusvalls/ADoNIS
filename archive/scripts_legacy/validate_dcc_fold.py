"""Validate the spectral-function fold (diffpi/dcc_fold.py): elementary sigma(W,Q^2)
folded over the 12C spectral function + electron acceptance -> nuclear dsigma/dW,
compared QUANTITATIVELY to the oracle.

Gates:
  1. SHAPE    -- folded nuclear dsigma/dW matches the oracle better than the bare
     free-nucleon curve (peak alignment + Fermi broadening of the FWHM).
  2. GRADIENT -- d(dsigma/dW)/d(pw_norm[p33]) via reweighting matches finite
     differences (exact kind-1; the spectral sampling is detached).
  3. PHYSICS  -- the fold broadens the elementary resonance (FWHM_folded > FWHM_free).

Run:  python validate_dcc_fold.py
"""
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffpi.dcc import DCCKnobs
from diffpi.dcc_xsec import DCCCrossSection
from diffpi.dcc_fold import fold_dsigma_dW

ok = True


def chk(name, cond, extra=""):
    global ok; ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {extra}")


oracle = np.load("oracle/oracle_distributions.npz")
W_edges = jnp.asarray(oracle["W_edges"])
dW_or = oracle["dsig_W"]
Wc = 0.5 * (np.asarray(W_edges)[:-1] + np.asarray(W_edges)[1:])
xs = DCCCrossSection()


def unit(v):
    a = np.trapz(v, Wc)
    return v / a if a > 0 else v


def fwhm(v):
    a = np.where(v >= 0.5 * v.max())[0]
    return float(Wc[a[-1]] - Wc[a[0]]) if a.size else np.nan


# folded nuclear curve
h_fold = np.asarray(fold_dsigma_dW(DCCKnobs(), W_edges, jax.random.PRNGKey(1), n=500000))
# bare free-nucleon curve (step-4 style) at the oracle's Q^2 support, on the same W grid
Q2_grid = jnp.linspace(float(oracle["Q2_edges"][0]), float(oracle["Q2_edges"][-1]), 41)
h_free = np.asarray(xs.dsigma_dW(jnp.asarray(Wc), Q2_grid))

sm, so, sf = unit(h_fold), unit(dW_or), unit(h_free)
L1_fold = float(np.sum(np.abs(sm - so)) * np.mean(np.diff(Wc)))
L1_free = float(np.sum(np.abs(sf - so)) * np.mean(np.diff(Wc)))
peak_m, peak_o = Wc[np.argmax(sm)], Wc[np.argmax(so)]
chk("folded peak aligns with oracle (<=40 MeV)", abs(peak_m - peak_o) <= 40,
    f"(folded={peak_m:.0f}, oracle={peak_o:.0f} MeV)")
chk("fold improves shape match vs bare free-nucleon", L1_fold < L1_free,
    f"(L1 fold={L1_fold:.3f} < free={L1_free:.3f})")
chk("fold broadens the resonance (Fermi)", fwhm(sm) > fwhm(sf),
    f"(FWHM folded={fwhm(sm):.0f} > free={fwhm(sf):.0f}; oracle={fwhm(so):.0f} MeV)")

# ---- Gate 2: exact reweighting gradient through the fold ---------------------- #
i33 = xs.D.labels.index("p33")
n_pw = len(xs.D.labels)


def scalar(delta):
    pw = tuple(delta if i == i33 else 0.0 for i in range(n_pw))
    h = fold_dsigma_dW(DCCKnobs(pw_norm=pw), W_edges, jax.random.PRNGKey(7), n=120000)
    return jnp.sum(h)             # same key -> CRN, so FD is clean


g = float(jax.grad(scalar)(0.0))
eps = 1e-3
fd = float((scalar(eps) - scalar(-eps)) / (2 * eps))
chk("d/d(pw_norm[p33]) == finite diff (exact reweighting)",
    np.isclose(g, fd, rtol=1e-4), f"(auto={g:.5e}, fd={fd:.5e})")

# ---- figure ------------------------------------------------------------------ #
fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
ax.plot(Wc, so, "o-", ms=4, color="C1", label=f"oracle (nuclear)  FWHM {fwhm(so):.0f}")
ax.plot(Wc, sm, "-", lw=2, color="C0", label=f"folded model  FWHM {fwhm(sm):.0f}  (L1 {L1_fold:.3f})")
ax.plot(Wc, sf, "--", lw=1.5, color="C3", label=f"bare free-nucleon  FWHM {fwhm(sf):.0f}  (L1 {L1_free:.3f})")
ax.axvline(1232, ls=":", color="k", lw=0.8)
ax.set_xlabel("W [MeV]"); ax.set_ylabel("normalised  d$\\sigma$/dW")
ax.set_title("Step 5: spectral-function fold vs oracle  (e + $^{12}$C, unit area)")
ax.legend()
fig.tight_layout()
fig.savefig("dcc_fold_validation.png", dpi=110)
print("\nsaved -> dcc_fold_validation.png")
print(f"\nDCC spectral fold: {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
