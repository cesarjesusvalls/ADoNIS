"""Fast contract tests for the shared ratio panel (adonis.workflow.plotting.chi2_ratio_panel):
histogram mode ({values,w}) and curve mode ({x,y,yerr}), on Agg axes.  Pins the returned
dict(chi2, ndf, ach_ado): identical ref/ado -> chi2==0, ach_ado==1; a known offset scales chi2."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adonis.workflow.plotting import chi2_ratio_panel


def _axes():
    fig, (a0, a1) = plt.subplots(2, 1)
    return fig, a0, a1


def test_hist_identical_is_zero_chi2():
    edges = np.linspace(0.0, 10.0, 6)
    vals = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
    ref = {"values": vals, "w": np.ones_like(vals)}
    ado = {"values": vals.copy(), "w": np.ones_like(vals)}
    fig, a0, a1 = _axes()
    r = chi2_ratio_panel(a0, a1, edges, ref, ado, label="x")
    plt.close(fig)
    assert abs(r["chi2"]) < 1e-9
    assert r["ndf"] > 0
    assert abs(r["ach_ado"] - 1.0) < 1e-9


def test_hist_offset_raises_chi2():
    edges = np.linspace(0.0, 10.0, 6)
    vals = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
    ref = {"values": vals, "w": np.ones_like(vals)}
    ado = {"values": vals, "w": 2.0 * np.ones_like(vals)}
    fig, a0, a1 = _axes()
    r = chi2_ratio_panel(a0, a1, edges, ref, ado, label="x")
    plt.close(fig)
    assert r["chi2"] > 0.0
    assert r["ach_ado"] < 1.0


def test_curve_mode_contract():
    x = np.array([1.0, 2.0, 3.0, 4.0]); y = np.array([1.0, 2.0, 3.0, 4.0]); ye = 0.1 * np.ones(4)
    ref = {"x": x, "y": y, "yerr": ye}
    ado = {"x": x, "y": y.copy(), "yerr": ye.copy()}
    fig, a0, a1 = _axes()
    r = chi2_ratio_panel(a0, a1, x, ref, ado, label="c")
    plt.close(fig)
    assert {"chi2", "ndf", "ach_ado"} <= set(r)
    assert abs(r["chi2"]) < 1e-9 and abs(r["ach_ado"] - 1.0) < 1e-9
