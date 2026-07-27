"""Validate the exclusive ACHILLES DCC RES hadron current (amp_dcc_sl.f amplitude, irot_q=1) vs
instrumented ACHILLES (RESDUMP).  After the single universal normalisation, the per-event amps2
must match ACHILLES with small scatter -- the angular/frame/isospin assembly is reproduced; the
residual is the amplitude-table interpolation (spline vs interpolate_amp).  The constant is
CHANNEL-INDEPENDENT (pi+ vs pi0)."""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from adonis.channels.dcc.current import exclusive_amps2

DUMP = Path(__file__).resolve().parent / "data" / "res_dump_achilles.txt"


def _parse():
    rows = []
    for l in DUMP.read_text().splitlines():
        g = lambda k: float(re.search(k + r"=([-+0-9.eE]+)", l).group(1))
        v4 = lambda k: [float(x) for x in re.search(k + r"=([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+)", l).groups()]
        gi = lambda k: int(re.search(k + r"=(-?\d+)", l).group(1))
        rows.append(dict(li=[g("liE")] + v4("li"), lo=[g("loE")] + v4("lo"),
                         hi=[g("hiE")] + v4("hi"), hiID=gi("hiID"),
                         hN=[g("hNE")] + v4("hN"), hP=[g("hPE")] + v4("hP"),
                         hPID=gi("hPID"), amps2=g("amps2"),
                         flux=g("flux"), initwgt=g("initwgt"), psw=g("psw")))
    return rows


def test_res_current_matches_achilles():
    rows = _parse()
    r_all, r_pip, r_pi0, my_l, ach_l, w_l = [], [], [], [], [], []
    for r in rows:
        if r["amps2"] <= 0:
            continue
        itiz = 1 if r["hiID"] == 2212 else -1
        my = float(exclusive_amps2(r["li"], r["lo"], r["hi"], r["hN"], r["hP"], itiz, r["hPID"]))
        if my <= 0:
            continue
        ratio = my / r["amps2"]
        r_all.append(ratio)
        (r_pip if r["hPID"] == 211 else r_pi0).append(ratio)
        my_l.append(my); ach_l.append(r["amps2"])
        w_l.append(r["flux"] * r["initwgt"] * 0.5 * r["psw"])    # per-event xsec weight (sans amps2)
    r_all = np.array(r_all); my_l = np.array(my_l); ach_l = np.array(ach_l); w_l = np.array(w_l)
    assert len(r_all) > 50
    # absolute scale correct (normalised): mean ratio ~1
    assert abs(r_all.mean() - 1.0) < 0.05, r_all.mean()
    # per-event scatter small (angular/frame assembly bit-exact; residual = interpolation)
    assert r_all.std() / r_all.mean() < 0.06, r_all.std() / r_all.mean()
    # channel-independent
    assert abs(np.mean(r_pip) - np.mean(r_pi0)) / np.mean(r_all) < 0.02
    # NO large per-event outliers (the spline-stencil off-by-one under-shot the low-W/low-Q^2
    # edge by up to 25% -- invisible to mean/std, but it biased sigma_RES by ~27%).
    assert np.max(np.abs(r_all - 1.0)) < 0.03, f"max amps2 dev {np.max(np.abs(r_all-1)):.3f}"
    # the SIGMA-weighted ratio (what actually sets sigma_RES) must be ~1
    sig_ratio = np.sum(my_l * w_l) / np.sum(ach_l * w_l)
    assert abs(sig_ratio - 1.0) < 0.02, f"xsec-weighted amps2 ratio {sig_ratio:.4f}"


if __name__ == "__main__":
    test_res_current_matches_achilles()
    print("exclusive RES current matches ACHILLES (constant ratio, channel-independent)")
