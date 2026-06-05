"""Validation of the full differentiable final state (fold_final_state) vs the ACHILLES
oracle, across the full set of final-state observables.

Two figures in figures/:
  * final_state_consistency.png -- the un-integration is unbiased: the full-final-state
    fold reproduces the angle-integrated fold's dsigma/dW, dsigma/dQ2 (flat ratio ~1).
  * final_state_oracle.png      -- the new lab/CM observables (cos theta*_pi, phi*_pi,
    |p_pi|, lepton E/angle, recoil-nucleon |p|, W, Q2) vs the ACHILLES oracle, with
    per-bin statistical errors and a model/oracle ratio panel + chi2/ndf.

Comparison binning is an EXACT integer coarsening of the oracle's fine binning (group
G fine bins by reshape-sum), so the model and oracle share identical edges -- avoids the
rebinning aliasing that a center-assignment regroup introduces when the fine:coarse ratio
is non-integer.

Oracle: oracle/oracle_finalstate.npz (make_oracle_finalstate.py).  Model: chunked
fold_final_state generation (bounded memory) with per-chunk progress.
Run:  python validate_final_state.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adonis.primary.dcc.amplitudes import DCCKnobs
from adonis.primary.dcc.structure import HadronStructure
from adonis.primary.dcc.fold_integrated import fold_full_events
from adonis.primary.dcc.channel import fold_final_state
from adonis import observables as obs

os.makedirs("figures", exist_ok=True)
CHUNK, NCHUNK = 50_000, 40                     # 2.0M events (match the 2M oracle)
ORACLE = "oracle/oracle_finalstate.npz"
hs = HadronStructure(n_theta=16, n_phi=16, spline=False)

od = np.load(ORACLE)
N_OR = int(od["n_events"])

# group factor G per observable: must divide the number of fine oracle bins exactly, so
# the coarse comparison edges = fine_edges[::G] and oracle coarse = reshape(-1,G).sum(1).
GROUP = {"W": 6, "Q2": 5, "cos_theta_star": 5, "ppi_mag": 5, "phi_star": 2,
         "lepton_energy": 5, "lepton_costheta": 5, "nucleon_mom": 6}
SCALE = {"Q2": 1e-6}            # x-axis display scale (others 1.0)

EDGES, O_SW, O_SW2 = {}, {}, {}
for name, G in GROUP.items():
    fe = od[f"edges_{name}"]
    nfine = len(fe) - 1
    assert nfine % G == 0, f"{name}: {nfine} fine bins not divisible by {G}"
    EDGES[name] = fe[::G]
    O_SW[name] = od[f"sw_{name}"].reshape(-1, G).sum(1)
    O_SW2[name] = od[f"sw2_{name}"].reshape(-1, G).sum(1)

# model accumulators (binned on the SAME EDGES) + integrated-fold ref for W,Q2
acc = {f"{p}_{k}": np.zeros(len(EDGES[k]) - 1) for k in EDGES for p in ("m", "m2")}
acci = {f"{p}_{k}": np.zeros(len(EDGES[k]) - 1) for k in ("W", "Q2") for p in ("i", "i2")}
sigw_m = sigw_i = 0.0


def add(accd, pre, name, v, w):
    e = EDGES[name]
    accd[f"{pre}_{name}"] += np.histogram(v, bins=e, weights=w)[0]
    accd[f"{pre}2_{name}"] += np.histogram(v, bins=e, weights=w ** 2)[0]


# Cache the accumulated model/integrated histograms so COSMETIC plot edits don't re-pay
# the ~1-min generation.  Re-generate only when the cache is missing or FS_REGEN=1 is set
# (or the binning/stats changed -> cache key mismatch).
CACHE = "figures/_fs_val_cache.npz"
KEY = f"{CHUNK}x{NCHUNK}_" + "_".join(f"{k}{GROUP[k]}" for k in sorted(GROUP))


def _generate():
    global sigw_m, sigw_i
    for c in range(NCHUNK):
        Wf, Q2f, wf = map(np.asarray, fold_full_events(DCCKnobs(), jax.random.PRNGKey(1000 + c), n=CHUNK, hs=hs))
        add(acci, "i", "W", Wf, wf); add(acci, "i", "Q2", Q2f, wf); sigw_i += wf.sum()
        ev = fold_final_state(DCCKnobs(), jax.random.PRNGKey(5000 + c), n=CHUNK, hs=hs)
        w = np.asarray(ev.w); sigw_m += w.sum()
        for name in EDGES:
            add(acc, "m", name, np.asarray(obs.OBSERVABLES[name](ev)), w)
        print(f"  chunk {c+1}/{NCHUNK}: int kept {int((wf!=0).sum())}  FS kept {int((w!=0).sum())}", flush=True)
    np.savez(CACHE, key=KEY, sigw_m=sigw_m, sigw_i=sigw_i, **acc, **acci)
    print(f"cached model histograms -> {CACHE}")


if os.environ.get("FS_REGEN") or not os.path.exists(CACHE) or str(np.load(CACHE)["key"]) != KEY:
    _generate()
else:
    d = np.load(CACHE)
    for k in acc: acc[k] = d[k]
    for k in acci: acci[k] = d[k]
    sigw_m, sigw_i = float(d["sigw_m"]), float(d["sigw_i"])
    print(f"loaded cached model histograms <- {CACHE}  (FS_REGEN=1 to regenerate)")
print(f"sigma ratio (FS/int) = {sigw_m / sigw_i:.4f}")


def norm(h, h2, edges):
    dx, S = np.diff(edges), h.sum()
    return h / dx / S, np.sqrt(h2) / dx / S


# ---------------- figure 1: consistency vs the integrated fold --------------- #
def panel_cmp(ax, axr, name, title, xlabel):
    edges = EDGES[name]; sc = SCALE.get(name, 1.0); x = 0.5 * (edges[:-1] + edges[1:]) * sc
    di, ei = norm(acci[f"i_{name}"], acci[f"i2_{name}"], edges)
    dm, em = norm(acc[f"m_{name}"], acc[f"m2_{name}"], edges)
    ax.errorbar(x, di / sc, yerr=ei / sc, fmt="o", ms=3, color="darkorange", capsize=2, lw=1, label="angle-integrated")
    ax.errorbar(x, dm / sc, yerr=em / sc, fmt="s-", ms=2.5, color="cornflowerblue", capsize=2, lw=1, label="full final state")
    ax.set_ylabel("norm. d$\\sigma$"); ax.set_title(title); ax.legend(fontsize=7)
    R = np.divide(dm, di, out=np.full_like(dm, np.nan), where=di > 0)
    axr.axhline(1, color="0.5", lw=0.8); axr.plot(x, R, "o", ms=3, color="cornflowerblue")
    axr.set_ylim(0.85, 1.15); axr.set_xlabel(xlabel); axr.set_ylabel("FS / int")


fig1 = plt.figure(figsize=(11, 4.5))
gs = fig1.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.05, wspace=0.25)
panel_cmp(fig1.add_subplot(gs[0, 0]), fig1.add_subplot(gs[1, 0]), "W", "d$\\sigma$/dW", "W [MeV]")
panel_cmp(fig1.add_subplot(gs[0, 1]), fig1.add_subplot(gs[1, 1]), "Q2", "d$\\sigma$/d$Q^2$", "$Q^2$ [GeV$^2$]")
fig1.suptitle(f"Un-integration unbiased: full-final-state vs angle-integrated fold ($\\sigma$ ratio {sigw_m/sigw_i:.3f})")
fig1.savefig("figures/final_state_consistency.png", dpi=120, bbox_inches="tight")
print("saved -> figures/final_state_consistency.png")

# ---------------- figure 2: new observables vs the ACHILLES oracle ----------- #
PANELS = [("W", "d$\\sigma$/dW", "W [MeV]"),
          ("Q2", "d$\\sigma$/d$Q^2$", "$Q^2$ [GeV$^2$]"),
          ("cos_theta_star", r"d$\sigma$/d$\cos\theta^*_\pi$", r"$\cos\theta^*_\pi$ (piN-CM vs q)"),
          ("ppi_mag", r"d$\sigma$/d$|p_\pi|$", r"$|p_\pi|$ [MeV]"),
          ("phi_star", r"d$\sigma$/d$\phi^*_\pi$", r"$\phi^*_\pi$ [rad]"),
          ("lepton_energy", r"d$\sigma$/d$E_\ell$", r"$E_\ell$ [MeV]"),
          ("lepton_costheta", r"d$\sigma$/d$\cos\theta_\ell$", r"$\cos\theta_\ell$ (lab)"),
          ("nucleon_mom", r"d$\sigma$/d$|p_N|$", r"$|p_N|$ [MeV]")]

fig2 = plt.figure(figsize=(16, 9))
# 2 macro-rows x 4 cols with a generous gap between rows; each cell holds a tightly-
# coupled (panel, ratio) pair via a nested 2x1 subgridspec that shares the x-axis.
outer = fig2.add_gridspec(2, 4, hspace=0.32, wspace=0.30)
slots = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]

for (name, title, xlabel), (r, col) in zip(PANELS, slots):
    edges = EDGES[name]; sc = SCALE.get(name, 1.0); x = 0.5 * (edges[:-1] + edges[1:]) * sc
    do, eo = norm(O_SW[name], np.maximum(O_SW2[name], 0), edges)
    dm, em = norm(acc[f"m_{name}"], acc[f"m2_{name}"], edges)
    inner = outer[r, col].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig2.add_subplot(inner[0])
    axr = fig2.add_subplot(inner[1], sharex=ax)               # share x with its panel
    ax.errorbar(x, do / sc, yerr=eo / sc, fmt="o", ms=3, color="darkorange", capsize=2, lw=1,
                label=f"ACHILLES ({N_OR/1e6:.1f}M)")
    ax.errorbar(x, dm / sc, yerr=em / sc, fmt="s-", ms=2.5, color="cornflowerblue", capsize=2, lw=1,
                label=f"ADoNIS ({CHUNK*NCHUNK//1000}k)")
    ax.set_ylabel("norm. d$\\sigma$"); ax.set_title(title); ax.legend(fontsize=7)
    ax.tick_params(labelbottom=False)                         # x ticks only on the ratio
    R = np.divide(dm, do, out=np.full_like(dm, np.nan), where=do > 0)
    rel = np.sqrt(np.divide(em, dm, out=np.zeros_like(em), where=dm > 0) ** 2
                  + np.divide(eo, do, out=np.zeros_like(eo), where=do > 0) ** 2)
    Re = R * rel
    good = np.isfinite(R) & (Re > 0)
    chi2 = float(np.sum(((R[good] - 1) / Re[good]) ** 2)); ndf = int(good.sum())
    axr.axhline(1, color="0.5", lw=0.8)
    axr.errorbar(x, R, yerr=Re, fmt="o", ms=3, color="cornflowerblue", capsize=2, lw=1)
    axr.set_ylim(0.8, 1.2); axr.set_xlabel(xlabel); axr.set_ylabel("FS/orcl")
    axr.text(0.02, 0.06, f"$\\chi^2$/ndf={chi2:.0f}/{ndf}={chi2/max(ndf,1):.2f}",
             transform=axr.transAxes, fontsize=8, va="bottom")
    print(f"{name:16s} chi2/ndf = {chi2:.1f}/{ndf} = {chi2/max(ndf,1):.2f}")

fig2.savefig("figures/final_state_oracle.png", dpi=110, bbox_inches="tight")
print("saved -> figures/final_state_oracle.png")
