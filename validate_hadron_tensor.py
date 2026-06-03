"""Validate the angular primitives of the full hadron-tensor port (milestone 6b):
Wigner-d (setdfun/fblmmx), Clebsch-Gordan (cbg), associated Legendre (ylmsub).

These are the knob-INDEPENDENT building blocks of the faithful amp_dcc_sl.f
assembly; validated here against closed-form values so that when 6c assembles them
into zj_mu, any residual disagreement is in the assembly/contraction, not the
primitives.

Run:  python validate_hadron_tensor.py
"""
import numpy as np

from diffpi.hadron_tensor import wigner_d_half, wigner_d_int, cbg, legendre_ylm

ok = True


def chk(name, a, b, atol=1e-10):
    global ok
    p = np.isclose(a, b, atol=atol)
    ok = ok and bool(p)
    print(f"  [{'PASS' if p else 'FAIL'}] {name:30s} port={a:+.6f}  ref={b:+.6f}")


th = 0.7
x, c, s = np.cos(th), np.cos(th / 2), np.sin(th / 2)

print("Wigner-d (half-integer J, 2*projection indexing):")
chk("d^1/2_{1/2,1/2}", wigner_d_half(1, 1, 1, x), c)
chk("d^1/2_{1/2,-1/2}", wigner_d_half(1, 1, -1, x), -s)
chk("d^3/2_{3/2,3/2}", wigner_d_half(3, 3, 3, x), c ** 3)
chk("d^3/2_{3/2,1/2}", wigner_d_half(3, 3, 1, x), -np.sqrt(3) * c ** 2 * s)
chk("d^3/2_{1/2,1/2}", wigner_d_half(3, 1, 1, x), c * (3 * c ** 2 - 2))
chk("d^5/2_{5/2,5/2}", wigner_d_half(5, 5, 5, x), c ** 5)
print("Wigner-d (integer):")
chk("d^1_{0,0}", wigner_d_int(1, 0, 0, c, s), np.cos(th))
chk("d^2_{0,0}", wigner_d_int(2, 0, 0, c, s), (3 * x ** 2 - 1) / 2)
chk("d^2_{1,0}", wigner_d_int(2, 1, 0, c, s), -np.sqrt(1.5) * np.sin(th) * np.cos(th))

print("Associated Legendre / Y_l^m(theta,0):")
sin_th = np.sin(th)                 # full-angle sine (NOT the half-angle s above)
bl, L = legendre_ylm(4, x), 4
chk("Y_0^0", bl[0, L], np.sqrt(1 / (4 * np.pi)))
chk("Y_1^0", bl[1, L], np.sqrt(3 / (4 * np.pi)) * x)
chk("Y_1^1", bl[1, 1 + L], -np.sqrt(3 / (8 * np.pi)) * sin_th)
chk("Y_2^0", bl[2, L], np.sqrt(5 / (16 * np.pi)) * (3 * x ** 2 - 1))
chk("Y_2^2", bl[2, 2 + L], np.sqrt(15 / (32 * np.pi)) * sin_th ** 2)

print("Clebsch-Gordan (hand-checked + isospin-relevant):")
chk("<1/2,1/2;1/2,-1/2|0,0>", cbg(.5, .5, .5, -.5, 0, 0), 1 / np.sqrt(2))
chk("<1,0;1,0|2,0>", cbg(1, 0, 1, 0, 2, 0), np.sqrt(2 / 3))
chk("<1,1;1/2,-1/2|3/2,1/2>", cbg(1, 1, .5, -.5, 1.5, .5), np.sqrt(1 / 3))
chk("<1,1;1/2,-1/2|1/2,1/2>", cbg(1, 1, .5, -.5, .5, .5), np.sqrt(2 / 3))

# exhaustive cross-check vs sympy if available
try:
    from sympy.physics.quantum.cg import CG
    from sympy import S
    from fractions import Fraction as F
    n, bad = 0, 0
    for j1 in [0.5, 1, 1.5]:
        for j2 in [0.5, 1]:
            for m1 in np.arange(-j1, j1 + 1):
                for m2 in np.arange(-j2, j2 + 1):
                    for J in np.arange(abs(j1 - j2), j1 + j2 + 1):
                        M = m1 + m2
                        if abs(M) > J:
                            continue
                        a = cbg(j1, m1, j2, m2, J, M)
                        b = float(CG(S(F(j1)), S(F(m1)), S(F(j2)), S(F(m2)),
                                     S(F(J)), S(F(M))).doit())
                        n += 1
                        bad += not np.isclose(a, b, atol=1e-9)
    chk(f"exhaustive CG vs sympy ({n} cases)", bad, 0)
except ImportError:
    print("  (sympy unavailable -- skipped exhaustive CG cross-check)")

print(f"\nHadron-tensor angular primitives (6b): {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
