"""Validate the differentiable DCC cross-section assembly (diffpi/dcc_xsec.py).

Gates:
  1. PHYSICS  -- elementary sigma(W) at fixed Q^2 peaks at the Delta(1232) with a
     physical width (the assembly reproduces the resonance from the real amplitudes).
  2. GRADIENT -- d/d(pw_norm) of a scalar functional matches forward-mode jacfwd
     and finite differences EXACTLY (pure kind-1 / deterministic, no MC noise).
  3. KNOB     -- pw_norm[p33] scales the Delta peak by (1+delta)^2 (bilinear);
     the axial knob leaves the electromagnetic (vec) cross section unchanged.
  4. ORACLE   -- qualitative overlay: elementary dsigma/dW (at the oracle's mean
     Q^2) vs the oracle dsigma/dW. The Delta peak alignment is the test; the full
     quantitative match needs the spectral-function fold (next milestone).

Run:  python validate_dcc_xsec.py
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

X = DCCCrossSection()
labels = X.D.labels
i33 = labels.index("p33")
ok = True


def chk(name, cond, extra=""):
    global ok; ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {extra}")


# ---- Gate 1: Delta(1232) peak ------------------------------------------------ #
W_grid = jnp.linspace(1080.0, 1500.0, 211)
Q2_fix = 100000.0   # 0.1 GeV^2
sigW = jax.vmap(lambda w: X.sigma(w, Q2_fix))(W_grid)
W_peak = float(W_grid[int(jnp.argmax(sigW))])
# FWHM
half = 0.5 * float(jnp.max(sigW))
above = np.where(np.asarray(sigW) >= half)[0]
fwhm = float(W_grid[above[-1]] - W_grid[above[0]]) if above.size else np.nan
chk("Delta peak in 1200-1260 MeV", 1200 <= W_peak <= 1260, f"(peak={W_peak:.0f} MeV)")
chk("physical width 80-180 MeV", 80 <= fwhm <= 180, f"(FWHM={fwhm:.0f} MeV)")

# ---- Gate 2: exact gradient w.r.t. pw_norm ----------------------------------- #
n_pw = len(labels)


def functional(delta_vec):
    """Scalar functional of a pw_norm perturbation: integral of sigma over (W,Q2)."""
    pw = tuple(delta_vec[i] for i in range(n_pw))
    Wg = jnp.linspace(1100.0, 1400.0, 61)
    Qg = jnp.linspace(0.0, 300000.0, 31)
    return jnp.sum(X.dsigma_dW(Wg, Qg, DCCKnobs(pw_norm=pw)))


d0 = jnp.zeros(n_pw)
g_auto = jax.grad(functional)(d0)
g_fwd = jax.jacfwd(functional)(d0)
chk("grad == jacfwd (autodiff consistency)",
    np.allclose(np.asarray(g_auto), np.asarray(g_fwd), rtol=1e-8))
# finite-difference the p33 component
eps = 1e-3
e33 = d0.at[i33].set(eps)
fd = float((functional(e33) - functional(-e33)) / (2 * eps))
chk("grad[p33] == finite diff (exact, no MC noise)",
    np.isclose(float(g_auto[i33]), fd, rtol=1e-5), f"(auto={float(g_auto[i33]):.4e}, fd={fd:.4e})")

# ---- Gate 3: knob response --------------------------------------------------- #
sig_nom = float(X.sigma(1232.0, Q2_fix))
delta = 0.10
pw = tuple(delta if i == i33 else 0.0 for i in range(n_pw))
sig_p33 = float(X.sigma(1232.0, Q2_fix, DCCKnobs(pw_norm=pw)))
# at the peak P33 dominates; scaling its amplitude by (1+delta) scales |A|^2 ~ (1+delta)^2.
# the other waves are unscaled, so the ratio is between 1 and (1+delta)^2.
ratio = sig_p33 / sig_nom
chk("pw_norm[p33] raises Delta-peak xsec toward (1.1)^2",
    1.10 < ratio < 1.21, f"(ratio={ratio:.4f}, (1.1)^2={1.1**2:.3f})")
# axial knob must NOT change the electromagnetic (vec) cross section
sig_ax = float(X.sigma(1232.0, Q2_fix, DCCKnobs(axial_strength=3.0), current="vec"))
chk("axial_strength leaves vec xsec unchanged", np.isclose(sig_ax, sig_nom, rtol=1e-12))

# ---- Gate 4: oracle overlay -------------------------------------------------- #
oracle = np.load(Path("oracle/oracle_distributions.npz"))
W_edges = oracle["W_edges"]; dsig_W_oracle = oracle["dsig_W"]
W_cen = 0.5 * (W_edges[:-1] + W_edges[1:])
# mean Q^2 of the oracle (for the fixed-Q^2 elementary curve)
Q2_edges = oracle["Q2_edges"]; dsig_Q2 = oracle["dsig_Q2"]
Q2_cen = 0.5 * (Q2_edges[:-1] + Q2_edges[1:])
Q2_mean = float(np.sum(Q2_cen * dsig_Q2) / np.sum(dsig_Q2))
# model dsigma/dW integrated over the oracle's Q^2 support
Q2_grid = jnp.linspace(float(Q2_edges[0]), float(Q2_edges[-1]), 41)
W_model = jnp.asarray(W_cen)
dsigW_model = np.asarray(X.dsigma_dW(W_model, Q2_grid))
# normalise both to unit area over the compared range for a SHAPE comparison
def unit(v, x):
    a = np.trapz(v, x); return v / a if a > 0 else v
shape_model = unit(dsigW_model, W_cen)
shape_oracle = unit(dsig_W_oracle, W_cen)
# peak alignment
wp_model = W_cen[int(np.argmax(shape_model))]
wp_oracle = W_cen[int(np.argmax(shape_oracle))]
chk("model & oracle dsigma/dW peaks within 60 MeV",
    abs(wp_model - wp_oracle) <= 60, f"(model={wp_model:.0f}, oracle={wp_oracle:.0f} MeV)")

# ---- figure ------------------------------------------------------------------ #
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
ax[0].plot(np.asarray(W_grid), np.asarray(sigW), color="C3")
ax[0].axvline(1232, ls="--", color="k", lw=0.8); ax[0].axvline(W_peak, ls=":", color="C3")
ax[0].set_title(f"elementary $\\sigma(W)$ @ $Q^2$=0.1 GeV$^2$\npeak {W_peak:.0f} MeV, FWHM {fwhm:.0f} MeV")
ax[0].set_xlabel("W [MeV]"); ax[0].set_ylabel(r"$\sigma$ [arb]")

# per-PW decomposition at the peak
pw_contrib = np.asarray(X.sigma_per_pw(1232.0, Q2_fix))
order = np.argsort(pw_contrib)[::-1]
ax[1].bar([labels[i] for i in order], pw_contrib[order] / pw_contrib.sum(), color="C0")
ax[1].set_title("partial-wave fraction at W=1232")
ax[1].set_ylabel("fraction"); ax[1].tick_params(axis="x", rotation=60)

ax[2].plot(W_cen, shape_oracle, "o-", ms=3, label="oracle (nuclear)", color="C1")
ax[2].plot(W_cen, shape_model, "-", label="model (free-nucleon)", color="C3")
ax[2].axvline(1232, ls="--", color="k", lw=0.8)
ax[2].set_title("$d\\sigma/dW$ shape overlay (unit area)")
ax[2].set_xlabel("W [MeV]"); ax[2].set_ylabel("normalised"); ax[2].legend()
fig.tight_layout()
out = Path("dcc_xsec_validation.png")
fig.savefig(out, dpi=110)
print(f"\nsaved -> {out}")
print(f"\nDCC cross-section assembly: {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
