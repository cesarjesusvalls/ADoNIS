"""Validate the exclusive-DCC frame/angular primitives (boost + half-integer Wigner-d) against
analytic values -- the foundation for the irot_q=1 exclusive RES current port."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from adonis.channels.dcc.wigner import boost_matrix, setdfun


def test_boost_to_rest_and_roundtrip():
    tot = np.array([1473.8, 120.0, -200.0, 300.0])
    xlr = boost_matrix(tot, to_cm=True)
    cm = xlr @ tot
    assert np.max(np.abs(cm[1:])) < 1e-6
    back = boost_matrix(tot, to_cm=False) @ (xlr @ tot)
    assert np.max(np.abs(back - tot)) < 1e-9


def test_wigner_d_analytic():
    th = 0.7; x = np.cos(th); c = np.cos(th / 2); s = np.sin(th / 2)
    dfun, off = setdfun(x, 5)
    d = lambda lx, mf, mi: dfun[lx, mf + off, mi + off]
    assert abs(d(1, 1, 1) - c) < 1e-12
    assert abs(d(1, 1, -1) + s) < 1e-12
    assert abs(d(3, 3, 3) - c ** 3) < 1e-12
    assert abs(d(3, 3, -3) + s ** 3) < 1e-12
    assert abs(d(3, 1, 1) - (3 * np.cos(th) - 1) / 2 * c) < 1e-12


if __name__ == "__main__":
    test_boost_to_rest_and_roundtrip(); test_wigner_d_analytic()
    print("DCC kinematics primitives OK")
