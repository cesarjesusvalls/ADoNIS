"""Validate the differentiable form-factor port (diffpi/form_factors.py) and the
M_A reweight knob wired into the DCC cross section.

Gates:
  1. PORT     -- z-expansion reproduces F_A(0) = -g_A (built-in normalisation),
     dipole matches its closed form, both fall off with Q^2.
  2. GRADIENT -- dF_A/dM_A (dipole) and d/d(params) (z-exp) match finite diff.
  3. KNOB     -- larger M_A => harder axial form factor (more high-Q^2 strength);
     the axial reweight is exactly 1 at M_A nominal; vec xsec is untouched, the
     axial xsec hardens in Q^2 with M_A.
  4. PHYSICS  -- z-expansion and dipole(M_A=1) agree near Q^2=0 and both ~ -1.27.

Run:  python validate_form_factors.py
"""
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffpi.form_factors import (axial_dipole, axial_zexpansion, axial_zexpansion_vec,
                                 axial_reweight_dipole, MA_NOMINAL, GAN1, CC_PARAMS)
from diffpi.dcc import DCCKnobs
from diffpi.dcc_xsec import DCCCrossSection

ok = True


def chk(name, cond, extra=""):
    global ok; ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {extra}")


# ---- Gate 1 & 4: port correctness ------------------------------------------- #
FA0_z = float(axial_zexpansion(0.0))
FA0_d = float(axial_dipole(0.0))
# the z-expansion is an independent fit with its own effective g_A (~1.276),
# within ~1% of the dipole's 1.2694 -- not constrained to be identical.
chk("z-expansion F_A(0) ~ -g_A within 1%", abs(FA0_z / -GAN1 - 1) < 0.01,
    f"(z-exp={FA0_z:.4f}, -g_A={-GAN1:.4f}, dev {FA0_z/-GAN1-1:+.2%})")
chk("dipole F_A(0) = -g_A exactly", np.isclose(FA0_d, -GAN1, atol=1e-9), f"({FA0_d:.4f})")
# both fall off with Q^2
chk("dipole falls off", float(axial_dipole(1.0)) > FA0_d and abs(axial_dipole(1.0)) < abs(FA0_d))
chk("z-exp falls off", abs(float(axial_zexpansion(1.0))) < abs(FA0_z))
# z-exp and dipole agree to ~20% over 0-0.5 GeV^2 (different params, same g_A & slope scale)
Q2g = jnp.linspace(0, 0.5, 6)
zz = np.asarray(axial_zexpansion_vec(Q2g)); dd = np.asarray(axial_dipole(Q2g))
chk("z-exp ~ dipole(M_A=1) within 20% to 0.5 GeV^2", np.all(np.abs(zz / dd - 1) < 0.20),
    f"(max dev {np.max(np.abs(zz/dd-1)):.2%})")

# ---- Gate 2: exact gradients ------------------------------------------------ #
gMA = float(jax.grad(lambda ma: axial_dipole(0.3, ma))(1.0))
fd = float((axial_dipole(0.3, 1.0 + 1e-5) - axial_dipole(0.3, 1.0 - 1e-5)) / 2e-5)
chk("dF_A/dM_A (dipole) == finite diff", np.isclose(gMA, fd, rtol=1e-5),
    f"(auto={gMA:.4e}, fd={fd:.4e})")
# z-exp grad wrt a coefficient
def zexp_p(a0):
    p = (a0,) + tuple(CC_PARAMS[1:])
    return axial_zexpansion(0.2, p)
ga0 = float(jax.grad(zexp_p)(CC_PARAMS[0]))
fda0 = float((zexp_p(CC_PARAMS[0] + 1e-5) - zexp_p(CC_PARAMS[0] - 1e-5)) / 2e-5)
chk("d/d(z-coeff a0) == finite diff", np.isclose(ga0, fda0, rtol=1e-5))

# ---- Gate 3: M_A reweight + cross-section response --------------------------- #
chk("axial reweight == 1 at M_A nominal",
    np.isclose(float(axial_reweight_dipole(200000.0, MA_NOMINAL)), 1.0, atol=1e-12))
# larger M_A -> harder FF -> reweight > 1 at high Q^2
r_hi = float(axial_reweight_dipole(300000.0, 1.2))   # 0.3 GeV^2, M_A=1.2
chk("larger M_A hardens FF (reweight>1 at high Q^2)", r_hi > 1.0, f"(r={r_hi:.3f})")

X = DCCCrossSection()
Q2_lo, Q2_hi = 50000.0, 400000.0   # 0.05, 0.4 GeV^2
# vec xsec must be independent of M_A
v0 = float(X.sigma(1232.0, Q2_hi, DCCKnobs(), "vec"))
v1 = float(X.sigma(1232.0, Q2_hi, DCCKnobs(axial_MA=1.3), "vec"))
chk("M_A leaves vec (EM) xsec unchanged", np.isclose(v0, v1, rtol=1e-12))
# axial xsec ratio hi/lo Q^2 increases with M_A (spectrum hardens)
def hardness(MA):
    a_lo = float(X.sigma(1232.0, Q2_lo, DCCKnobs(axial_MA=MA), "axial"))
    a_hi = float(X.sigma(1232.0, Q2_hi, DCCKnobs(axial_MA=MA), "axial"))
    return a_hi / a_lo
chk("axial Q^2 spectrum hardens with M_A", hardness(1.3) > hardness(1.0),
    f"(hi/lo: MA1.0={hardness(1.0):.3f}, MA1.3={hardness(1.3):.3f})")

# ---- figure ------------------------------------------------------------------ #
Q2 = jnp.linspace(0, 1.5, 200)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for MA, c in [(0.8, "C0"), (1.0, "k"), (1.2, "C3")]:
    ax[0].plot(np.asarray(Q2), np.asarray(axial_dipole(Q2, MA)),
               color=c, label=f"dipole $M_A$={MA}")
ax[0].plot(np.asarray(Q2), np.asarray(axial_zexpansion_vec(Q2)), "--", color="C2",
           label="z-expansion (ACHILLES default)")
ax[0].set_xlabel(r"$Q^2$ [GeV$^2$]"); ax[0].set_ylabel(r"$F_A(Q^2)$")
ax[0].set_title("axial form factor"); ax[0].legend()

Q2M = jnp.linspace(0, 600000.0, 200)
for MA, c in [(0.8, "C0"), (1.0, "k"), (1.2, "C3")]:
    r = np.asarray(jax.vmap(lambda q: axial_reweight_dipole(q, MA))(Q2M)) ** 2
    ax[1].plot(np.asarray(Q2M) / 1e6, r, color=c, label=f"$M_A$={MA}")
ax[1].axhline(1.0, ls=":", color="gray")
ax[1].set_xlabel(r"$Q^2$ [GeV$^2$]"); ax[1].set_ylabel(r"axial $|A|^2$ reweight")
ax[1].set_title("M_A reweight applied to the DCC axial block"); ax[1].legend()
fig.tight_layout()
fig.savefig("form_factor_validation.png", dpi=110)
print("\nsaved -> form_factor_validation.png")
print(f"\nForm-factor port + M_A knob: {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
