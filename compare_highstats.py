"""Compare the faithful full-port fold to the neutrino oracle at a configurable bin
resolution, with PROPER per-bin statistical errors propagated to the ratio.

Both sides are weighted Monte-Carlo, so the per-bin statistical error is the weighted
standard error sigma = sqrt(sum_i w_i^2) (NOT sqrt(N)). Both are stored EVENT-LEVEL
(oracle_nu_events.npz = 200k ACHILLES events; model_nu_events.npz = diffpi fold, run
make_model_events.py first), so we re-bin both identically at any NBINS. The unit-area
ratio R = d_model/d_oracle carries  sigma_R/R = sqrt((sigma_m/raw_m)^2+(sigma_o/raw_o)^2).

Run:  python make_model_events.py   # once, to produce model_nu_events.npz
      python compare_highstats.py   # bins + plots (set NBINS below)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NBINS = 50
W_RANGE = (1080.0, 1655.0)
Q2_RANGE = (0.0, 1.58e6)

oe = np.load("oracle/oracle_nu_events.npz")
me = np.load("model_nu_events.npz")
Wo, Q2o, wo = oe["W"], oe["Q2"], oe["w_nb"]
Wm, Q2m, wm = me["W"], me["Q2"], me["w"]
print(f"oracle: {len(wo):,} events   model: {len(wm):,} events")

We = np.linspace(*W_RANGE, NBINS + 1); Wc = 0.5 * (We[:-1] + We[1:])
Q2e = np.linspace(*Q2_RANGE, NBINS + 1); Q2c = 0.5 * (Q2e[:-1] + Q2e[1:])


def hist_sw(v, w, edges):
    raw = np.histogram(v, bins=edges, weights=w)[0]
    sw2 = np.histogram(v, bins=edges, weights=w ** 2)[0]
    return raw, np.sqrt(sw2)


def normalize(raw, err, edges):
    dx = np.diff(edges); S = raw.sum()
    rel = np.divide(err, raw, out=np.zeros_like(err), where=raw > 0)
    dens = raw / dx / S
    return dens, dens * rel


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


Mlbl = f"diffpi fold ({len(wm)/1e6:.1f}M evt)"
Olbl = f"ACHILLES oracle ({len(wo)/1e3:.0f}k evt)"
fig, ax = plt.subplots(2, 2, figsize=(12, 6.5), sharex="col",
                       gridspec_kw={"height_ratios": [3, 1], "hspace": 0.04})


def panel(col, vo, vm, edges, x, xlabel, ylabel, title, scale=1.0):
    o_raw, o_err = hist_sw(vo, wo, edges)
    m_raw, m_err = hist_sw(vm, wm, edges)
    do, edo = normalize(o_raw, o_err, edges)
    dm, edm = normalize(m_raw, m_err, edges)
    R, Re = ratio_with_err(m_raw, m_err, o_raw, o_err)
    c2, ndf = chi2_ndf(R, Re)
    top, bot = ax[0, col], ax[1, col]
    # SAME x (bin centres) for top and ratio; markers on both so alignment is explicit
    top.errorbar(x, do * scale, yerr=edo * scale, fmt="o", ms=3.5, color="C1",
                 capsize=2, lw=1, label=Olbl, zorder=3)
    top.errorbar(x, dm * scale, yerr=edm * scale, fmt="s-", ms=2.5, color="C3",
                 capsize=2, lw=1.3, label=Mlbl, zorder=2)
    top.set_ylabel(ylabel); top.set_title(title); top.legend()
    bot.axhline(1.0, color="0.5", lw=0.8)
    bot.errorbar(x, R, yerr=Re, fmt="o", ms=3.5, color="C3", capsize=2, lw=1)
    bot.set_ylim(0.85, 1.15); bot.set_xlabel(xlabel)
    bot.set_ylabel("model / oracle")
    bot.text(0.02, 0.07, f"$\\chi^2$/ndf = {c2:.0f}/{ndf} = {c2/ndf:.2f}",
             transform=bot.transAxes, fontsize=9, va="bottom")
    print(f"{title}: chi2/ndf = {c2:.1f}/{ndf} = {c2/ndf:.2f}")


panel(0, Wo, Wm, We, Wc, "W [MeV]", "norm d$\\sigma$/dW", "d$\\sigma$/dW")
panel(1, Q2o, Q2m, Q2e, Q2c / 1e6, r"$Q^2$ [GeV$^2$]", "norm d$\\sigma$/d$Q^2$",
      "d$\\sigma$/d$Q^2$", scale=1e6)
fig.suptitle(f"diffpi fold vs ACHILLES neutrino oracle -- {NBINS} bins, stat. errors on ratio")
fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.08, wspace=0.18)
fig.savefig("fold_full_highstats.png", dpi=120)
print("saved -> fold_full_highstats.png")
