"""Fit the ADoNIS CC0pi-Np STV prediction (differentiable event-bank reweight) to the GENIE
AR23_20i fake dataset, and read the parameter pulls against the KNOWN GENIE-vs-ACHILLES physics.

ADoNIS model  = exact per-event bank reweight w(theta) (bank_reweight.weight_jit), binned over the
                topological CC0pi-Np signal in the T2K delta_pT / delta_alphaT binning -> dsigma/dx,
                differentiable in the fitted knobs (autodiff, not FD).
Target        = GENIE fake dsigma/dx (scripts/altgen/build_fakedata.py), with the real T2K CC0pi-STV
                covariance adopted as the measurement error model (central values = GENIE).
Fitted knobs  = QE+RES sector, chosen to map onto the three known differences:
   M_A                 QE axial Q^2 shape        (GENIE z-expansion axial FF vs ACHILLES dipole)
   qe_norm             QE normalisation          (GENIE SuSAv2/CFG QE vs ACHILLES SF)
   kF_sf               initial-state Fermi scale (SF vs LFG -> delta_pT peak width)
   s_NN_el             common NN-elastic scale   (GENIE hN2018 vs ACHILLES INC -> delta_pT tail;
                       scales s_NN_elastic[pp/pn/nn] together — NOTE the legacy `sscat` knob is
                       IGNORED by bank_reweight (line 35 hardcodes 1.0), the split knobs are live)
   sabs                pion absorption strength  (RES->CC0pi feed: GENIE hN abs vs ACHILLES Oset)
   res_norm            RES normalisation         (GENIE Berger-Sehgal vs ACHILLES DCC)

Reports: nominal vs BFP chi2/ndf (absolute + shape), pulls with Hessian errors, and a figure.
Blueprint: mirrors tune.py (covariance chi2, Adam-to-BFP, Hessian param covariance).
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from analysis.t2k.differentiability import bank_plot as BP, bank_reweight as BR
from analysis.t2k.differentiability.full_knobs import nominal_knobs

BANKDIR = os.environ.get("ADONIS_EVENT_BANK", "output/event_bank")
FAKE = os.environ.get("ADONIS_FAKEDATA", "output/altgen/fakedata_ccqeres.npz")
OBS = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "dpt"
ABS = "--abs" in sys.argv                                    # absolute (default: shape/profiled-A)
assert OBS in ("dpt", "dat")

FIT = ["M_A_qe", "qe_norm", "kF_sf", "s_NN_el", "sabs", "res_norm"]   # M_A split post-30d12e6
# clip ranges (physical-ish); Eb_shift/f_NN_cex not fit here
LO = jnp.array([0.5, 0.3, 0.6, 0.3, 0.3, 0.3])
HI = jnp.array([2.0, 3.0, 1.6, 3.0, 3.0, 3.0])
CONV = (1e-33 / 12.0 * 1000.0 * 1e38) if OBS == "dpt" else (1e-33 / 12.0 * 1e38)


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

    B = BP.load_bank(BANKDIR); JB = BR.to_jax(B); grids = BR.default_grids(); nom = nominal_knobs()
    log(f"bank {len(B['w0'])} events")

    # ---- CC0pi-Np topological signal + delta_pT/alphaT bin index (STATIC) ----------------------- #
    mask_np, lead = BP.signal_cc0pi(B, topological=True)
    val = BP.dpt(B, lead) if OBS == "dpt" else BP.dat(B, lead)

    d = np.load(FAKE, allow_pickle=True)
    EDGES = d[f"{OBS}_edges"]; DATA = jnp.asarray(d[f"{OBS}_dsig"]); COV = np.asarray(d[f"{OBS}_t2k_cov"])
    # bin widths in NATIVE edge units (MeV for dpt, rad for dat) — CONV already carries the
    # MeV->GeV/c factor for dpt (mirrors tune.py); dividing by GeV widths here double-counts x1000.
    nb = len(DATA); bw = np.diff(EDGES)
    COVINV = jnp.asarray(np.linalg.inv(COV + 1e-12 * np.eye(nb)))
    binidx = jnp.asarray(np.clip(np.searchsorted(EDGES, val) - 1, 0, nb - 1))
    maskj = jnp.asarray(mask_np.astype(np.float64))
    bwj = jnp.asarray(bw)
    log(f"signal {int(mask_np.sum())} events; target {OBS} from {FAKE} "
        f"(GENIE chan frac QE/RES/MEC/COH={np.asarray(d['chan_frac'])})")

    def knobs_from(theta):
        k = dict(nom)
        for i, key in enumerate(FIT):
            if key == "s_NN_el":                       # common scale on the 3 NN-elastic channels
                k["s_NN_elastic"] = theta[i] * jnp.ones(3)
            else:
                k[key] = theta[i]
        return k

    @jax.jit
    def model_hist(theta):
        w = BR.weight_jit(JB, knobs_from(theta), grids) * maskj
        h = jax.ops.segment_sum(w, binidx, num_segments=nb)
        return h / bwj * CONV

    th0 = jnp.array([1.0 if k == "s_NN_el" else float(np.asarray(nom[k])) for k in FIT])
    nomh = model_hist(th0)
    A0 = float((nomh @ COVINV @ DATA) / (nomh @ COVINV @ nomh))

    def chi2_abs(theta):
        r = model_hist(theta) - DATA
        return r @ COVINV @ r

    def chi2_shape(theta):
        m = model_hist(theta)
        A = (m @ COVINV @ DATA) / (m @ COVINV @ m)
        r = A * m - DATA
        return r @ COVINV @ r

    chi2 = chi2_abs if ABS else chi2_shape
    ndf_nom = nb if ABS else nb - 1
    ndf_fit = nb - len(FIT) if ABS else nb - len(FIT) - 1
    tag = "abs" if ABS else "shape"
    c0 = float(chi2(th0))
    log(f"metric={tag}  NOMINAL chi2={c0:.1f}  chi2/ndf={c0/ndf_nom:.2f}  (A_nom={A0:.3f})")

    vg = jax.jit(jax.value_and_grad(chi2))
    # warm start (e.g. finalize a converged-but-killed run): ALTGEN_START="v1,v2,..." + ALTGEN_NIT=0
    start = os.environ.get("ALTGEN_START")
    theta = jnp.asarray([float(x) for x in start.split(",")]) if start else th0
    m = jnp.zeros(len(FIT)); v = jnp.zeros(len(FIT)); lr = 0.03
    NIT = int(os.environ.get("ALTGEN_NIT", "300")); traj = [np.asarray(theta)]
    for it in range(NIT):
        l, g = vg(theta)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
        mh = m / (1 - 0.9 ** (it + 1)); vh = v / (1 - 0.999 ** (it + 1))
        theta = jnp.clip(theta - lr * mh / (jnp.sqrt(vh) + 1e-8), LO, HI); traj.append(np.asarray(theta))
        if it % 40 == 0 or it == NIT - 1:
            log(f"  it {it:3d}  chi2/ndf={float(l)/ndf_fit:.3f}  " +
                " ".join(f"{k}={float(theta[i]):.3f}" for i, k in enumerate(FIT)))
    bfp = np.asarray(theta); cbf = float(chi2(theta))
    # Hessian = FD of the autodiff gradient (13 cheap grad evals). jax.hessian over the full-bank
    # graph (2.7GB captured constants) gets OOM-killed on this 16GB machine — the gradient graph is
    # fine, so central-difference it instead.
    # pinv: near-degenerate knob combos (e.g. M_A<->qe_norm on one observable) make H rank-deficient;
    # pinv reports errors in the identifiable subspace (blueprint caveat: invalid at clip rails).
    gfun = jax.jit(jax.grad(chi2))
    eps = 1e-4
    H = np.zeros((len(FIT), len(FIT)))
    for i in range(len(FIT)):
        e = jnp.zeros(len(FIT)).at[i].set(eps)
        H[i] = (np.asarray(gfun(theta + e)) - np.asarray(gfun(theta - e))) / (2 * eps)
    H = 0.5 * (H + H.T)
    V = 2.0 * np.linalg.pinv(H, rcond=1e-10)
    sig = np.sqrt(np.abs(np.diag(V)))
    ew = np.linalg.eigvalsh(H)
    if ew.min() < 1e-8 * ew.max():
        log(f"WARNING: Hessian near-singular (eig ratio {ew.min()/ew.max():.1e}) — degenerate direction; "
            f"errors are subspace-projected")
    railed = [(abs(bfp[i] - float(LO[i])) < 1e-3 or abs(bfp[i] - float(HI[i])) < 1e-3) for i in range(len(FIT))]
    log(f"BFP chi2={cbf:.1f}  chi2/ndf={cbf/ndf_fit:.2f}")

    print(f"\n==== ADoNIS QE+RES fit to GENIE AR23 fake {OBS} ({tag}; "
          f"nominal chi2/ndf={c0/ndf_nom:.2f} -> BFP {cbf/ndf_fit:.2f}) ====")
    nomvals = np.asarray(th0)
    print(f"{'knob':>20} {'nominal':>8} {'BFP':>8} {'+/-':>7} {'pull(sigma)':>11} {'flag':>6}")
    for i, k in enumerate(FIT):
        n = nomvals[i]; pull = (bfp[i] - n) / sig[i] if sig[i] > 0 else 0.0
        print(f"{k:>20} {n:8.3f} {bfp[i]:8.3f} {sig[i]:7.3f} {pull:11.2f} {'RAIL' if railed[i] else '':>6}")

    # ---- figure ---------------------------------------------------------------------------------- #
    ctr = 0.5 * (EDGES[1:] + EDGES[:-1]) / (1000.0 if OBS == "dpt" else 1.0)
    xed = EDGES / (1000.0 if OBS == "dpt" else 1.0)
    derr = np.sqrt(np.diag(COV))
    mnom = np.asarray(nomh); mbf = np.asarray(model_hist(theta))
    if not ABS:
        mnom = A0 * mnom
        Abf = float((model_hist(theta) @ COVINV @ DATA) / (model_hist(theta) @ COVINV @ model_hist(theta)))
        mbf = Abf * mbf
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].errorbar(ctr, np.asarray(DATA), yerr=derr, fmt="o", color="k", capsize=3,
                   label="GENIE AR23 fake data (+T2K cov)")
    ax[0].step(xed, np.append(mnom, mnom[-1]), where="post", color="0.5", ls=":", lw=1.5,
               label=f"ADoNIS nominal ({tag} chi2/ndf {c0/ndf_nom:.1f})")
    ax[0].step(xed, np.append(mbf, mbf[-1]), where="post", color="C0", lw=2,
               label=f"ADoNIS best fit ({tag} chi2/ndf {cbf/ndf_fit:.1f})")
    XL = r"$\delta p_T$ [GeV/c]" if OBS == "dpt" else r"$\delta\alpha_T$ [rad]"
    ax[0].set(xlabel=XL, title=f"ADoNIS QE+RES fit to GENIE AR23 fake data ({OBS}, {tag})")
    ax[0].legend(fontsize=8); ax[0].set_ylim(bottom=0)
    pulls = [(bfp[i] - nomvals[i]) / sig[i] if sig[i] > 0 else 0.0 for i, k in enumerate(FIT)]
    cols = ["C3" if abs(p) > 2 else "C0" for p in pulls]
    ax[1].barh(FIT[::-1], pulls[::-1], color=cols[::-1])
    ax[1].axvline(0, color="k", lw=0.8); ax[1].axvline(2, ls="--", color="0.6"); ax[1].axvline(-2, ls="--", color="0.6")
    ax[1].set(xlabel="pull = (BFP - nominal)/sigma", title="QE+RES knob pulls")
    for i, (k, p) in enumerate(zip(FIT[::-1], pulls[::-1])):
        ax[1].text(p, i, f"  {bfp[len(FIT)-1-i]:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    os.makedirs("output/figures", exist_ok=True)
    out = f"output/figures/altgen_fit_{OBS}_{tag}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    np.savez(f"output/altgen/fit_{OBS}_{tag}.npz", obs=OBS, tag=tag, fit=FIT, bfp=bfp, sig=sig,
             pulls=np.array(pulls), chi2_nom=c0, chi2_bf=cbf, ndf_nom=ndf_nom, ndf_fit=ndf_fit,
             traj=np.array(traj), edges=EDGES, data=np.asarray(DATA), derr=derr,
             model_nom=mnom, model_bf=mbf, A0=A0)
    log(f"[fig] {out}")


if __name__ == "__main__":
    main()
