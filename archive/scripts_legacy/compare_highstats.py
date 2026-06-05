"""Compare the faithful full-port fold to the neutrino oracle with PROPER per-bin
statistical errors propagated to the ratio.

Oracle: the accumulated high-statistics histograms (oracle/oracle_highstats.npz, sum w
and sum w^2 on a fine grid; run make_oracle_highstats.py) -- falls back to the 200k
event-level oracle if absent. Model: diffpi fold events (model_nu_events.npz, run
make_model_events.py). Both carry the weighted error sqrt(sum w^2); the fine oracle grid
is re-binned (grouped) to the comparison binning, and the model events are binned on the
SAME edges. The unit-area ratio R=d_m/d_o carries sigma_R/R=sqrt((s_m/r_m)^2+(s_o/r_o)^2).

Run:  python compare_highstats.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RW, RQ = 6, 4              # group this many fine oracle bins per comparison bin
ORACLE_NPZ = os.environ.get("ORACLE_NPZ", "oracle/oracle_highstats.npz")
OUT_PNG = os.environ.get("OUT_PNG", "fold_full_highstats.png")

me = np.load("model_nu_events.npz")
Wm, Q2m, wm = me["W"], me["Q2"], me["w"]


def group_edges(edges, r):
    n = (len(edges) - 1) // r * r
    return edges[: n + 1: r]


def regroup(h, r):
    n = len(h) // r * r
    return h[:n].reshape(-1, r).sum(1)


def hist_sw(v, w, edges):
    return (np.histogram(v, bins=edges, weights=w)[0],
            np.histogram(v, bins=edges, weights=w ** 2)[0])


# ---- oracle: high-stats accumulated histograms (preferred) or 200k events ----- #
if os.path.exists(ORACLE_NPZ):
    hsd = np.load(ORACLE_NPZ)
    N_ORACLE = int(hsd["n_events"])
    We = group_edges(hsd["We"], RW); Q2e = group_edges(hsd["Q2e"], RQ)
    oW_raw, oW_sw2 = regroup(hsd["swW"], RW), regroup(hsd["sw2W"], RW)
    oQ_raw, oQ_sw2 = regroup(hsd["swQ"], RQ), regroup(hsd["sw2Q"], RQ)
    osrc = "accum. hist"
else:
    oe = np.load("oracle/oracle_nu_events.npz")
    N_ORACLE = len(oe["w_nb"])
    We = np.linspace(1080.0, 1655.0, 51); Q2e = np.linspace(0.0, 1.58e6, 51)
    oW_raw, oW_sw2 = hist_sw(oe["W"], oe["w_nb"], We)
    oQ_raw, oQ_sw2 = hist_sw(oe["Q2"], oe["w_nb"], Q2e)
    osrc = "events"
oW_err, oQ_err = np.sqrt(oW_sw2), np.sqrt(oQ_sw2)
Wc = 0.5 * (We[:-1] + We[1:]); Q2c = 0.5 * (Q2e[:-1] + Q2e[1:])
print(f"oracle: {N_ORACLE:,} events ({osrc})   model: {len(wm):,} events")


def normalize(raw, err, edges):
    dx = np.diff(edges); S = raw.sum()
    rel = np.divide(err, raw, out=np.zeros_like(err), where=raw > 0)
    return raw / dx / S, (raw / dx / S) * rel


def ratio_with_err(m_raw, m_err, o_raw, o_err):
    Sm, So = m_raw.sum(), o_raw.sum()
    relm = np.divide(m_err, m_raw, out=np.zeros_like(m_err), where=m_raw > 0)
    relo = np.divide(o_err, o_raw, out=np.zeros_like(o_err), where=o_raw > 0)
    ok = (o_raw > 0) & (m_raw > 0)
    R = np.divide(m_raw / Sm, o_raw / So, out=np.full_like(m_raw, np.nan), where=ok)
    return R, R * np.sqrt(relm ** 2 + relo ** 2)


def chi2_ndf(R, Re):
    m = np.isfinite(R) & (Re > 0)
    return float(np.sum(((R[m] - 1) / Re[m]) ** 2)), int(m.sum())


Mlbl = f"diffpi fold ({len(wm)/1e6:.1f}M)"
Olbl = f"ACHILLES oracle ({N_ORACLE/1e6:.1f}M)" if N_ORACLE >= 1e6 else f"ACHILLES oracle ({N_ORACLE/1e3:.0f}k)"
fig, ax = plt.subplots(2, 2, figsize=(12, 6.5), sharex="col",
                       gridspec_kw={"height_ratios": [3, 1], "hspace": 0.04})


def panel(col, vm, edges, x, o_raw, o_err, xlabel, ylabel, title, scale=1.0):
    m_raw, m_sw2 = hist_sw(vm, wm, edges); m_err = np.sqrt(m_sw2)
    do, edo = normalize(o_raw, o_err, edges)
    dm, edm = normalize(m_raw, m_err, edges)
    R, Re = ratio_with_err(m_raw, m_err, o_raw, o_err)
    c2, ndf = chi2_ndf(R, Re)
    top, bot = ax[0, col], ax[1, col]
    top.errorbar(x, do * scale, yerr=edo * scale, fmt="o", ms=3.5, color="C1",
                 capsize=2, lw=1, label=Olbl, zorder=3)
    top.errorbar(x, dm * scale, yerr=edm * scale, fmt="s-", ms=2.5, color="C3",
                 capsize=2, lw=1.3, label=Mlbl, zorder=2)
    top.set_ylabel(ylabel); top.set_title(title); top.legend()
    bot.axhline(1.0, color="0.5", lw=0.8)
    bot.errorbar(x, R, yerr=Re, fmt="o", ms=3.5, color="C3", capsize=2, lw=1)
    bot.set_ylim(0.9, 1.1); bot.set_xlabel(xlabel); bot.set_ylabel("model / oracle")
    bot.text(0.02, 0.07, f"$\\chi^2$/ndf = {c2:.0f}/{ndf} = {c2/ndf:.2f}",
             transform=bot.transAxes, fontsize=9, va="bottom")
    print(f"{title}: chi2/ndf = {c2:.1f}/{ndf} = {c2/ndf:.2f}")


panel(0, Wm, We, Wc, oW_raw, oW_err, "W [MeV]", "norm d$\\sigma$/dW", "d$\\sigma$/dW")
panel(1, Q2m, Q2e, Q2c / 1e6, oQ_raw, oQ_err, r"$Q^2$ [GeV$^2$]",
      "norm d$\\sigma$/d$Q^2$", "d$\\sigma$/d$Q^2$", scale=1e6)
fig.suptitle("diffpi fold vs ACHILLES neutrino oracle -- statistical errors on ratio")
fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.08, wspace=0.18)
fig.savefig(OUT_PNG, dpi=120)
print(f"saved -> {OUT_PNG}")
