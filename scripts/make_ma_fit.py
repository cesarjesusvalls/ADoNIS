"""M_A closure fits on the differentiable ADoNIS predictions.

Generate ADoNIS "data" at a known M_A, then recover M_A by gradient descent two ways:
  (A) EXCLUSIVE fit  -- loss on the exclusive final-state distributions (cos theta*_pi,
      |p_pi|, E_lep);
  (B) INTEGRATED fit -- loss on the inclusive W and Q^2 distributions, obtained by
      integrating the SAME differentiable final state down to those marginals.

The point of (B): projecting/integrating the full final state to a W or Q^2 histogram is
just summing the differentiable per-event weights into bins keyed by the detached
observable -> gradients are preserved.  Both fits recover M_A_true, and grad(loss)
(autodiff) matches finite difference in both cases.

Kind-1 reweighting via sample_final_state/weight_from_sample: the proposal is sampled
ONCE per dataset; only the weight depends on M_A, so weight_from_sample is pure JAX ->
the loss is jit + grad friendly.  Two-replica unbiased loss (split the model sample in
half) avoids the model-variance bias of plain MSE.  Forward-mode (jvp) gradient: M_A is
scalar so it is as cheap as reverse mode but far lighter on memory.

Fit results are CACHED to figures/_ma_fit_cache.npz so plot-style edits re-render in ~1s
(set MA_REGEN=1, or change the stats, to refit).

Run:  python make_ma_fit.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adonis.primary.dcc.amplitudes import DCCKnobs
from adonis.primary.dcc.structure import HadronStructure
from adonis.primary.dcc.channel import sample_final_state, weight_from_sample
from adonis.core.event import EventRecord
from adonis import observables as obs
from adonis.core.autodiff import Adam

_T0 = time.time()
def log(msg):
    print(f"[{time.time()-_T0:6.1f}s] {msg}", flush=True)

os.makedirs("figures", exist_ok=True)
os.makedirs("data/cache", exist_ok=True)
hs = HadronStructure(spline=False)
MA_TRUE, MA_INIT = 1.20, 0.90
# stats overridable via env (e.g. ADONIS_NDATA=120000 ADONIS_NMODEL=40000 ADONIS_ITERS=80
# for a quick low-stats look); defaults are the high-stats configuration.
NDATA = int(os.environ.get("ADONIS_NDATA", 1_000_000))
NMODEL = int(os.environ.get("ADONIS_NMODEL", 100_000))
DATA_CHUNK = min(int(os.environ.get("ADONIS_DATA_CHUNK", 100_000)), NDATA)
ITERS = int(os.environ.get("ADONIS_ITERS", 200))
CACHE = "data/cache/_ma_fit_cache.npz"
CKEY = f"{NDATA}_{NMODEL}_{MA_TRUE}_{MA_INIT}_{ITERS}"

EDGES = {
    "W": np.linspace(1080, 1640, 29), "Q2": np.linspace(0, 1.6e6, 25),
    "cos_theta_star": np.linspace(-1, 1, 21), "ppi_mag": np.linspace(0, 700, 25),
    "lepton_energy": np.linspace(0, 1400, 25),
}
SETS = {"exclusive": ["cos_theta_star", "ppi_mag", "lepton_energy"],
        "integrated": ["W", "Q2"]}


def ev_of(S):
    z = jnp.zeros(S["n"])
    return EventRecord(k=S["k_lab"], kp=S["kp_lab"], p_struck=S["p_struck"], p_pi=S["p_pi"],
                       p_N=S["p_N"], w=z, channel=z, pid_pi=z, pid_N=z, pid_Ni=z,
                       W=S["W"], Q2_adj=S["Q2_adj"])


def bin_idx(S, name):
    e = jnp.asarray(EDGES[name]); nb = e.shape[0] - 1
    v = obs.OBSERVABLES[name](ev_of(S))
    return jax.lax.stop_gradient(jnp.clip(jnp.searchsorted(e, v) - 1, 0, nb - 1)), nb


# ---- model histograms / loss / fit (reference globals DATA, Sm, MIDX, NB, H) ------ #
def model_hist(MA, name, sl):
    w, _ = weight_from_sample(DCCKnobs(axial_MA=MA), Sm, use_spline=False)
    return jax.ops.segment_sum(w[sl], MIDX[name][sl], num_segments=NB[name]) / H


def disp_hist(MA, name):
    """Full-sample display histogram (per-generated-event normalised, like the data)."""
    w, _ = weight_from_sample(DCCKnobs(axial_MA=MA), Sm, use_spline=False)
    return np.asarray(jax.ops.segment_sum(w, MIDX[name], num_segments=NB[name])) / NMODEL


def make_loss(obs_names):
    def loss(MA):
        tot, ndf = 0.0, 0
        for name in obs_names:
            d, e = jnp.asarray(DATA[name][0]), jnp.asarray(DATA[name][1])
            mA = model_hist(MA, name, slice(0, H)); mB = model_hist(MA, name, slice(H, 2 * H))
            tot = tot + jnp.sum((mA - d) * (mB - d) / (e ** 2))   # unbiased two-replica chi2
            ndf += d.shape[0]
        return tot / ndf
    return loss


def fit(obs_names, lr=0.05, iters=ITERS, final_lr_frac=0.02):
    loss = make_loss(obs_names)
    vg = jax.jit(lambda MA: jax.jvp(loss, (MA,), (1.0,)))      # forward-mode value+grad
    log(f"[{'/'.join(obs_names)}] compiling forward value+grad (first call) ...")
    tc = time.time(); g_ad = float(vg(MA_INIT)[1])
    log(f"  compiled in {time.time()-tc:.1f}s; checking grad vs finite diff ...")
    eps = 5e-3
    g_fd = float((vg(MA_INIT + eps)[0] - vg(MA_INIT - eps)[0]) / (2 * eps))   # cheap (compiled)
    log(f"  grad@init AD={g_ad:.4e} FD={g_fd:.4e} "
        f"rel={abs(g_ad-g_fd)/(abs(g_ad)+abs(g_fd)+1e-30):.2e}")
    MA = jnp.asarray(MA_INIT); opt = Adam(lr); st = opt.init(MA)
    hist = [float(MA)]; losses = []; titer = time.time()
    for it in range(iters):
        opt.lr = lr * (final_lr_frac ** (it / max(iters - 1, 1)))   # anneal -> settle
        l, g = vg(MA); MA, st = opt.update(MA, jnp.asarray(g), st)
        hist.append(float(MA)); losses.append(float(l))
        if it % 30 == 0 or it == iters - 1:
            rate = (it + 1) / (time.time() - titer)
            log(f"  iter {it:3d}/{iters}  loss={float(l):.4e}  M_A={float(MA):.4f}  "
                f"({rate:.1f} it/s, ETA {(iters-it-1)/max(rate,1e-9):.0f}s)")
    MAf = float(MA)
    preds = {name: dict(data=DATA[name][0], derr=DATA[name][1],
                        init=disp_hist(MA_INIT, name), final=disp_hist(MAf, name))
             for name in obs_names}
    return dict(hist=np.array(hist), losses=np.array(losses), MAf=MAf, gad=g_ad, gfd=g_fd, preds=preds)


def run_fits():
    """Build the high-stats data target (chunked), sample the model once, run both fits,
    cache and return the results."""
    global DATA, Sm, MIDX, NB, H
    nchunks = NDATA // DATA_CHUNK
    log(f"generating DATA target at M_A={MA_TRUE}: {nchunks} x {DATA_CHUNK:,} = {NDATA:,} events ...")
    nb = {name: len(EDGES[name]) - 1 for name in EDGES}
    accw = {name: np.zeros(nb[name]) for name in EDGES}
    accw2 = {name: np.zeros(nb[name]) for name in EDGES}
    for c in range(nchunks):
        Sd = sample_final_state(jax.random.PRNGKey(101 + c), n=DATA_CHUNK, hs=hs)
        wd = np.asarray(weight_from_sample(DCCKnobs(axial_MA=MA_TRUE), Sd, use_spline=False)[0])
        for name in EDGES:
            idx = np.asarray(bin_idx(Sd, name)[0])
            accw[name] += np.bincount(idx, weights=wd, minlength=nb[name])
            accw2[name] += np.bincount(idx, weights=wd ** 2, minlength=nb[name])
        del Sd, wd
        log(f"  data chunk {c+1}/{nchunks}")
    DATA = {name: (accw[name] / NDATA, np.sqrt(accw2[name]) / NDATA + 1e-15) for name in EDGES}

    log(f"sampling MODEL proposal (N={NMODEL:,}) ...")
    Sm = sample_final_state(jax.random.PRNGKey(202), n=NMODEL, hs=hs)
    H = NMODEL // 2
    MIDX = {name: bin_idx(Sm, name)[0] for name in EDGES}
    NB = {name: bin_idx(Sm, name)[1] for name in EDGES}

    res = {}
    for tag, names in SETS.items():
        log(f"=== {tag} fit ({', '.join(names)}) ===")
        res[tag] = fit(names)
    np.savez(CACHE, key=CKEY, results=np.array(res, dtype=object))
    log(f"cached fit results -> {CACHE}")
    return res


# ---- load cache or run ----------------------------------------------------------- #
if not os.environ.get("MA_REGEN") and os.path.exists(CACHE) \
        and str(np.load(CACHE, allow_pickle=True)["key"]) == CKEY:
    results = np.load(CACHE, allow_pickle=True)["results"].item()
    log(f"loaded cached fit results <- {CACHE}  (MA_REGEN=1 to refit)")
else:
    results = run_fits()

# ---- plot (from results; restyle freely, re-runs in ~1s on the cache) ------------- #
XLAB = {"W": "W [MeV]", "Q2": "$Q^2$ [GeV$^2$]", "cos_theta_star": r"$\cos\theta^*_\pi$",
        "ppi_mag": r"$|p_\pi|$ [MeV]", "lepton_energy": r"$E_\ell$ [MeV]"}
SC = {"Q2": 1e-6}
ncol = 1 + max(len(v) for v in SETS.values())
fig, axes = plt.subplots(len(SETS), ncol, figsize=(4.2 * ncol, 4.0 * len(SETS)))
for row, (tag, R) in enumerate(results.items()):
    h, maf, preds = R["hist"], R["MAf"], R["preds"]
    axc = axes[row, 0]
    axc.axhline(MA_TRUE, color="darkorange", lw=1.5, ls="--", label=f"$M_A$ true = {MA_TRUE}")
    axc.plot(h, "-", color="cornflowerblue", lw=2, label=f"fit $\\to$ {maf:.4f}")
    axc.axhline(MA_INIT, color="0.6", lw=1, ls=":", label=f"init = {MA_INIT}")
    axc.set_xlabel("iteration"); axc.set_ylabel("$M_A$ [GeV]")
    axc.set_title(f"{tag}: $M_A$ convergence"); axc.legend(fontsize=8)
    axc.set_ylim(min(MA_INIT, MA_TRUE) - 0.08, max(MA_INIT, MA_TRUE) + 0.08)
    for j, name in enumerate(SETS[tag]):
        ax = axes[row, 1 + j]
        e = np.asarray(EDGES[name]); sc = SC.get(name, 1.0)
        x = 0.5 * (e[:-1] + e[1:]) * sc; dx = np.diff(e); p = preds[name]
        ax.errorbar(x, p["data"] / dx, yerr=p["derr"] / dx, fmt="o", ms=3, color="darkorange",
                    capsize=2, lw=1, label=f"data ($M_A$={MA_TRUE})", zorder=3)
        ax.plot(x, p["init"] / dx, ":", color="0.5", lw=1.5, label=f"init ($M_A$={MA_INIT})")
        ax.plot(x, p["final"] / dx, "-", color="cornflowerblue", lw=1.5, label=f"fit ($M_A$={maf:.3f})")
        ax.set_xlabel(XLAB[name]); ax.set_ylabel("d$\\sigma$/dx [a.u.]")
        ax.set_title(name); ax.legend(fontsize=7)
    for j in range(1 + len(SETS[tag]), ncol):
        axes[row, j].axis("off")
fig.suptitle("M$_A$ closure on differentiable ADoNIS predictions: convergence + data/initial/final", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("figures/ma_fit_closure.png", dpi=120, bbox_inches="tight")
print("saved -> figures/ma_fit_closure.png")
for tag in SETS:
    print(f"{tag:11s}: M_A_true={MA_TRUE}  ->  fit {results[tag]['MAf']:.4f}  "
          f"(grad AD/FD rel {abs(results[tag]['gad']-results[tag]['gfd'])/(abs(results[tag]['gad'])+abs(results[tag]['gfd'])+1e-30):.1e})")
