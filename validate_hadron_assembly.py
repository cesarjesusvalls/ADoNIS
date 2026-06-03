"""Milestone 6c/6d gate: the full hadron-tensor assembly (hadron_assembly.py).

Checks, at increasing strength:
  (A) machinery sanity -- W^{mu,nu} is Hermitian; the transverse response
      W_T = W11+W22 is real & positive; a longitudinal response W33 now EXISTS
      (it was identically absent from the transverse-only dcc_xsec).
  (B) Delta(1232) -- the transverse response peaks in the Delta region for the
      pure-I=3/2 p33 channel, as it must.
  (C) multipole structure -- decompose W_T(W) into per-partial-wave diagonal terms
      and the same-L interference remainder; the diagonal part is what dcc_xsec
      approximates, the remainder is the physics the full port restores.
  (D) differentiability -- d W_T / d(axial reweight) flows cleanly (finite, no NaN).

This is the regression gate before wiring the lepton-tensor contraction + fold.
Run:  python validate_hadron_assembly.py
"""
import numpy as np
import jax
import jax.numpy as jnp

from diffpi.dcc import DCCAmplitudes, DCCKnobs
from diffpi.dcc_loader import load_cached
from diffpi.hadron_assembly import (build_zmtx, angular_kernel, current_and_tensor,
                                     IGM1_LIST)

jax.config.update("jax_enable_x64", True)

M_N, M_PI = 938.272, 138.0
t = load_cached()
amp = DCCAmplitudes(t)
twoJ, twoL, twoI = np.asarray(t.pw_2J), np.asarray(t.pw_2L), np.asarray(t.pw_2I)

ok = True
def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# --- EM, proton, pi0-p channel (isovector I-CG; vector current only) -------- #
# tcrz=0 (EM), tiz=+1/2 (proton), tpiz=0 (pi0) -> tpinz=+1/2.
ker = angular_kernel(twoJ, twoL, twoI, tcrz=0.0, tiz=0.5, tpinz=0.5, tpiz=0.0,
                     tm_f=1.0, n_theta=16, n_phi=16)

Wgrid = np.linspace(1090.0, 1700.0, 80)
Q2 = 0.1e6                                  # 0.1 GeV^2

def hadron_W(Wval, Q2val, r_axial=None, mode=10):
    vec, isv, axial = amp.amplitudes(jnp.asarray(Wval), jnp.asarray(Q2val))
    zmtx = build_zmtx(vec, isv, axial, jnp.asarray(Wval), jnp.asarray(Q2val),
                      twoJ, twoL, twoI, mode=mode, itiz=1, m_N=M_N, m_pi=M_PI,
                      r_axial=r_axial)
    return current_and_tensor(zmtx, ker)

Wmats = [np.asarray(hadron_W(w, Q2)) for w in Wgrid]
Wmats = np.stack(Wmats)                      # (nW, 4, 4)
W_T = (Wmats[:, 1, 1] + Wmats[:, 2, 2])
W_L = Wmats[:, 3, 3]
W_00 = Wmats[:, 0, 0]

print("(A) machinery sanity")
herm = np.max(np.abs(Wmats - np.conj(np.transpose(Wmats, (0, 2, 1)))))
check("W^{mu,nu} Hermitian", herm < 1e-8, f"max|W-W^dag|={herm:.1e}")
check("W_T real", np.max(np.abs(W_T.imag)) < 1e-8, f"max|Im|={np.max(np.abs(W_T.imag)):.1e}")
check("W_T >= 0", np.all(W_T.real > -1e-12), f"min={W_T.real.min():.3e}")
check("longitudinal W_L exists", np.max(np.abs(W_L)) > 0,
      f"max|W33|={np.max(np.abs(W_L)):.3e}")

print("(B) Delta(1232) peak (transverse response)")
wpk = Wgrid[np.argmax(W_T.real)]
check("W_T peaks in Delta region", 1180 < wpk < 1300, f"peak at W={wpk:.0f} MeV")

print("(C) multipole decomposition (diagonal vs same-L interference)")
# diagonal: sum over single-pw hadron tensors (zero all other pw in zmtx)
def hadron_W_single(Wval, Q2val, ipw):
    vec, isv, axial = amp.amplitudes(jnp.asarray(Wval), jnp.asarray(Q2val))
    mask = jnp.zeros(vec.shape[1]).at[ipw].set(1.0)
    vec = vec * mask; isv = isv * mask; axial = axial * mask
    zmtx = build_zmtx(vec, isv, axial, jnp.asarray(Wval), jnp.asarray(Q2val),
                      twoJ, twoL, twoI, mode=10, itiz=1, m_N=M_N, m_pi=M_PI)
    return current_and_tensor(zmtx, ker)

iWpk = int(np.argmax(W_T.real))
Wpk = Wgrid[iWpk]
diag = sum(np.asarray(hadron_W_single(Wpk, Q2, i))[1, 1].real
           + np.asarray(hadron_W_single(Wpk, Q2, i))[2, 2].real
           for i in range(len(twoJ)))
full = W_T.real[iWpk]
interf = full - diag
check("diagonal multipole part dominates at peak",
      abs(diag) > 0 and abs(interf) < abs(diag),
      f"diag={diag:.3e} interf={interf:+.3e} ({100*interf/diag:+.1f}%)")

# compare W-SHAPE to the diagonal dcc_xsec vector response
from diffpi.dcc_xsec import DCCCrossSection
xs = DCCCrossSection(amp)
sig_diag = np.asarray([xs.sigma(jnp.asarray(w), jnp.asarray(Q2), DCCKnobs(),
                                current="vec") for w in Wgrid])
# normalise both to unit peak for shape comparison
sa, sb = W_T.real / W_T.real.max(), sig_diag / sig_diag.max()
shape_corr = np.corrcoef(sa, sb)[0, 1]
check("W-shape correlates with dcc_xsec vector response", shape_corr > 0.9,
      f"corr={shape_corr:.4f}")

print("(D) differentiability of the assembly")
def loss(logr):
    r = jnp.exp(logr)
    Wm = current_and_tensor(
        build_zmtx(*amp.amplitudes(jnp.asarray(1232.0), jnp.asarray(Q2)),
                   jnp.asarray(1232.0), jnp.asarray(Q2), twoJ, twoL, twoI,
                   mode=1, itiz=-1, m_N=M_N, m_pi=M_PI, r_axial=r), ker)
    return (Wm[1, 1] + Wm[2, 2]).real
g = jax.grad(loss)(0.0)
check("d W_T / d(log axial reweight) finite", np.isfinite(g), f"grad={g:.4e}")

# --- figure ----------------------------------------------------------------- #
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(Wgrid, W_T.real / W_T.real.max(), label="full W_T (transverse)")
    ax[0].plot(Wgrid, W_L.real / np.abs(W_L).max(), "--", label="W_L (longitudinal, NEW)")
    ax[0].plot(Wgrid, sig_diag / sig_diag.max(), ":", label="dcc_xsec diagonal (vec)")
    ax[0].axvline(1232, color="grey", lw=0.7); ax[0].set_xlabel("W [MeV]")
    ax[0].set_ylabel("response (unit peak)"); ax[0].legend(); ax[0].set_title("EM vector response")
    ax[1].plot(Wgrid, W_T.real, label="full (incl. same-L interference)")
    diag_W = np.array([sum(np.asarray(hadron_W_single(w, Q2, i))[1, 1].real
                           + np.asarray(hadron_W_single(w, Q2, i))[2, 2].real
                           for i in range(len(twoJ))) for w in Wgrid])
    ax[1].plot(Wgrid, diag_W, "--", label="diagonal multipoles only")
    ax[1].set_xlabel("W [MeV]"); ax[1].set_ylabel("W_T [arb]")
    ax[1].legend(); ax[1].set_title("multipole interference (full port restores it)")
    fig.tight_layout(); fig.savefig("hadron_assembly_validation.png", dpi=110)
    print("\nwrote hadron_assembly_validation.png")
except ImportError:
    pass

print(f"\nFull hadron-tensor assembly (6c/6d): {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
