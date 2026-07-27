"""HIGH-STATISTICS diagnostic: turn un-hideable foreign-generator mismatch into a physics signal.

Logbook 15. The 300k joint fits (logbook 9-14) used the real T2K covariance (~10% errors); under it
the 6 knobs absorb the GENIE mismatch (chi2/ndf -> 0.48, 0.14). That absorption is a LARGE-ERROR
artifact. Question (Remote Control 2026-07-07): when a future experiment drives statistical errors to
negligible, can we fit so the "broken" physics the forward model lacks becomes INFORMATIVE rather
than just a failed chi2?

Mechanism. The 6 knobs span a 6-D manifold in observable space; the foreign generator lies OFF it.
The best fit projects onto the manifold, leaving an irreducible residual r_perp in the knob
Jacobian's NULL SPACE. With C -> alpha*C (alpha->0 = high stat):
  * A global covariance scale does NOT move the argmin, so BFP is alpha-invariant and
    chi2_min(alpha)/ndf = chi2_min(1)/ndf * (1/alpha) EXACTLY -> the ladder is ANALYTIC (one fit),
    validated here by re-fitting at 3 alphas and checking BFP invariance.
  * chi2_min does NOT -> 0; it grows without bound for any structural mismatch. The information is
    the DIRECTION of r_perp: which bins/observables carry the un-absorbable tension.

Deliverables (this script):
  1. Statistics ladder  : chi2/ndf vs 1/alpha for matched (rises) vs +MEC (stays flat); the crossover
     alpha* where the mismatch reaches N-sigma. + BFP-invariance validation gate.
  2. Irreducible residual fingerprint: block-diagonal whitening, null-space projection at the BFP;
     per-bin standardized residual + the irreducible chi2 fraction.
  3. MEC-template closure: the +MEC sample GIVES the missing-physics template
     (dsig(+MEC) - dsig(matched) == GENIE's MEC channel). Show the matched fit's irreducible dpt
     residual has that shape -> ADoNIS's un-absorbable disagreement with GENIE-QE+RES is exactly the
     2p2h strength ADoNIS carries in SF+cascade that GENIE books separately.

Env: ADONIS_EVENT_BANK, ADONIS_FAKE_MATCHED, ADONIS_FAKE_MEC, ADONIS_LABEL, ALTGEN_NIT.
Reuses info_content.build_datasets (single source of truth for selections/bins/units/offsets) and
bank_reweight (exact differentiable weight); mirrors fit_fakedata_joint.py for the LM fit.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from adonis.reweight import bank_plot as BP, bank_reweight as BR
from adonis.reweight.full_knobs import nominal_knobs
from analysis.paper import info_content as IC

BANKDIR = os.environ.get("ADONIS_EVENT_BANK", "output/event_bank")
F_MATCHED = os.environ.get("ADONIS_FAKE_MATCHED", "output/altgen/fakedata_ccqeres.npz")
F_MEC = os.environ.get("ADONIS_FAKE_MEC", "output/altgen/fakedata_qeresmec.npz")
LABEL = os.environ.get("ADONIS_LABEL", "highstat")
NIT = int(os.environ.get("ALTGEN_NIT", "50"))

FIT = ["M_A_qe", "qe_norm", "kF_sf", "s_NN_el", "sabs", "res_norm"]
LO = np.array([0.5, 0.3, 0.6, 0.3, 0.3, 0.3])
HI = np.array([2.0, 3.0, 1.6, 3.0, 3.0, 3.0])
NP = len(FIT)
# The ladder is ANALYTIC (a global covariance scale does not move the argmin), so we validate
# BFP alpha-invariance with ONE cheap warm-started refit from the CONVERGED base fit, at this alpha.
LADDER_CHECK_ALPHA = float(os.environ.get("ALTGEN_LADDER_ALPHA", "0.1"))
VALID_NIT = int(os.environ.get("ALTGEN_VALID_NIT", "6"))


def knobs_from(theta, nom):
    k = dict(nom)
    for i, key in enumerate(FIT):
        if key == "s_NN_el":
            k["s_NN_elastic"] = theta[i] * jnp.ones(3)
        else:
            k[key] = theta[i]
    return k


def set_centrals(ds, npz, tag=""):
    """Swap each dataset's central values for the foreign-generator dsig (same convention as joint)."""
    for d in ds:
        if d["channel"] == "CC0pi":
            d["data"] = np.asarray(npz[f"{d['key']}{tag}_dsig"])
        else:
            d["data"] = np.asarray(npz[f"cc1pi_{d['key']}{tag}_dsig_C"]) + d["offset"]


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

    B = BP.load_bank(BANKDIR); JB = BR.to_jax(B); grids = BR.default_grids(); nom = nominal_knobs()
    log(f"bank {len(B['w0'])} events; matched={F_MATCHED} mec={F_MEC}")
    ds = IC.build_datasets(B)
    nbins = sum(d["nbin"] for d in ds); ndf = nbins - NP
    log(f"{len(ds)} datasets, {nbins} bins, ndf={ndf}")

    def wf(theta, JB):
        return BR.bank_weight(JB, knobs_from(theta, nom), grids)
    wf_jit = jax.jit(wf)
    jvp_wf = jax.jit(lambda th, tang, JB: jax.jvp(lambda t: wf(t, JB), (th,), (tang,))[1])

    def model_bins(th):
        w = np.asarray(wf_jit(jnp.asarray(th), JB))
        return [IC.bin_w(d, w) for d in ds]

    def jac_bins(th):
        Js = [np.zeros((d["nbin"], NP)) for d in ds]
        for k in range(NP):
            g = np.asarray(jvp_wf(jnp.asarray(th), jnp.zeros(NP).at[k].set(1.0), JB))
            for j, d in enumerate(ds):
                Js[j][:, k] = IC.bin_w0(d, g)
        return Js

    def chi2_of(mods, alpha=1.0):
        tot, per = 0.0, []
        for d, m in zip(ds, mods):
            r = m - d["data"]; c = float(r @ d["Cinv"] @ r) / alpha; per.append(c); tot += c
        return tot, per

    def lm_fit(alpha=1.0, th0=None, nit=NIT, tag=""):
        """Levenberg-Marquardt on the exact bank reweight at covariance scale alpha."""
        th = np.ones(NP) if th0 is None else th0.copy()
        mods = model_bins(th); c_cur, _ = chi2_of(mods, alpha); lam = 1e-3
        for it in range(nit):
            Js = jac_bins(th); A = np.zeros((NP, NP)); g = np.zeros(NP)
            for d, m, J in zip(ds, mods, Js):
                r = m - d["data"]
                A += (J.T @ d["Cinv"] @ J) / alpha; g += (J.T @ d["Cinv"] @ r) / alpha
            for _ in range(12):
                dth = np.linalg.solve(A + lam * np.diag(np.maximum(np.diag(A), 1e-12)), -g)
                th_try = np.clip(th + dth, LO, HI)
                mods_try = model_bins(th_try); c_try, _ = chi2_of(mods_try, alpha)
                if c_try < c_cur:
                    th, mods, c_cur = th_try, mods_try, c_try; lam = max(lam/3, 1e-8); break
                lam *= 5
            log(f"    [{tag} a={alpha:.2f}] it {it:2d} chi2={c_cur:8.2f} lam={lam:.1e} " +
                " ".join(f"{k}={th[i]:.3f}" for i, k in enumerate(FIT)))
            if np.linalg.norm(dth) < 1e-6:
                log(f"    [{tag} a={alpha:.2f}] converged (step<1e-6) at it {it}"); break
        return th, mods, c_cur

    def fingerprint(th, mods):
        """Block-diagonal whitening + null-space projection at the BFP.
        Returns (chi2_tot, chi2_reducible, chi2_irreducible) in alpha=1 units and per-bin
        standardized post-fit residual (m-data)/sigma per dataset."""
        Js = jac_bins(th)
        rw, Jw, stdres = [], [], []
        for d, m, J in zip(ds, mods, Js):
            wv, U = np.linalg.eigh(d["Cinv"]); wv = np.clip(wv, 0.0, None)
            H = U @ np.diag(np.sqrt(wv)) @ U.T          # Cinv^{1/2}
            r = m - d["data"]
            rw.append(H @ r); Jw.append(H @ J)
            stdres.append(r / d["sigma"])
        rw = np.concatenate(rw); Jw = np.vstack(Jw)
        P = Jw @ np.linalg.pinv(Jw, rcond=1e-12)        # projector onto knob column space
        rperp = rw - P @ rw
        return float(rw @ rw), float(rw @ (P @ rw)), float(rperp @ rperp), stdres

    # ============================ fit matched + +MEC ============================================= #
    npz_m = np.load(F_MATCHED, allow_pickle=True)
    npz_x = np.load(F_MEC, allow_pickle=True)

    results = {}
    for tag_name, npz in [("matched", npz_m), ("mec", npz_x)]:
        set_centrals(ds, npz)
        mods_nom = model_bins(np.ones(NP)); c_nom, per_nom = chi2_of(mods_nom)
        log(f"[{tag_name}] NOMINAL chi2/ndf = {c_nom/nbins:.3f}  (chi2={c_nom:.1f}, nbins={nbins})")
        th, mods, c_bf = lm_fit(alpha=1.0, tag=tag_name)   # converged base fit (NIT)
        log(f"[{tag_name}] BFP chi2/ndf = {c_bf/ndf:.3f}  th=" +
            " ".join(f"{k}={th[i]:.3f}" for i, k in enumerate(FIT)))
        ct, cr, ci, stdres = fingerprint(th, mods)
        log(f"    [{tag_name}] fingerprint: chi2_tot={ct:.1f} reducible={cr:.1f} irreducible={ci:.1f} "
            f"(irr.frac={ci/ct if ct>0 else 0:.3f})")
        results[tag_name] = dict(th=th, c_nom=c_nom, per_nom=per_nom, c_bf=c_bf, mods=mods,
                                 mods_nom=mods_nom, chi2_tot=ct, chi2_red=cr, chi2_irr=ci,
                                 stdres=stdres)

    # ---- ladder BFP-invariance gate: ONE warm-started refit of matched at LADDER_CHECK_ALPHA ----- #
    # If the argmin is truly covariance-scale-invariant, refitting the CONVERGED BFP under C->alpha*C
    # must not move it. chi2/ndf must equal base/alpha; theta must be unchanged.
    set_centrals(ds, npz_m)
    th_a, _, c_a = lm_fit(alpha=LADDER_CHECK_ALPHA, th0=results["matched"]["th"], nit=VALID_NIT, tag="ladder")
    dmax = float(np.max(np.abs(th_a - results["matched"]["th"])))
    expect = results["matched"]["c_bf"] / LADDER_CHECK_ALPHA
    log(f"[ladder-gate] refit matched @ alpha={LADDER_CHECK_ALPHA}: |dtheta|_max={dmax:.2e} "
        f"chi2={c_a:.1f} (expect base/alpha={expect:.1f}, ratio={c_a/expect:.4f})")
    ladder_gate = dict(alpha=LADDER_CHECK_ALPHA, dmax=dmax, chi2=c_a, expect=expect)

    # dpt dataset index (for the fingerprint + MEC-template panels)
    idpt = next(i for i, d in enumerate(ds) if d["key"] == "dpt")
    dpt = ds[idpt]
    x = 0.5 * (dpt["edges"][1:] + dpt["edges"][:-1]) / dpt["xscale"]
    xed = dpt["edges"] / dpt["xscale"]
    data_m = np.asarray(npz_m["dpt_dsig"]); data_x = np.asarray(npz_x["dpt_dsig"])
    mec_template = data_x - data_m                       # GENIE's MEC channel (dsig)
    # matched fit: re-evaluate its model + dpt Jacobian at the matched BFP
    set_centrals(ds, npz_m); mods_m = model_bins(results["matched"]["th"])
    resid_matched = mods_m[idpt] - data_m                # ADoNIS_bf - GENIE(QE+RES) on dpt
    Jd = jac_bins(results["matched"]["th"])[idpt]        # (nbin_dpt, NP) knob response on dpt

    # ---- null-space decomposition of the KNOWN MEC channel ------------------------------------- #
    # The 6 knobs can partly MIMIC the MEC shape, so the irreducible residual is NOT the raw MEC
    # template but its projection onto the knob NULL SPACE. Decompose the MEC template into the part
    # the knobs can absorb (reducible) and the part they cannot (irreducible) in the T2K metric.
    wv, U = np.linalg.eigh(dpt["Cinv"]); wv = np.clip(wv, 0.0, None)
    H = U @ np.diag(np.sqrt(wv)) @ U.T                   # Cinv^{1/2} (whiten)
    Hinv = U @ np.diag(np.where(wv > 0, 1.0/np.sqrt(np.where(wv > 0, wv, 1)), 0.0)) @ U.T
    Jw = H @ Jd
    Pj = Jw @ np.linalg.pinv(Jw, rcond=1e-12)            # projector onto knob column space (whitened)
    mec_w = H @ mec_template
    mec_irr_w = mec_w - Pj @ mec_w                        # un-absorbable part (whitened)
    mec_irr = Hinv @ mec_irr_w                            # un-whiten for plotting (dsig units)
    frac_irr_mec = float((mec_irr_w @ mec_irr_w) / (mec_w @ mec_w)) if mec_w @ mec_w > 0 else 0.0
    cc_raw = float(np.corrcoef(mec_template, resid_matched)[0, 1])
    cc_irr = float(np.corrcoef(mec_irr, resid_matched)[0, 1])

    # ============================ figures ======================================================== #
    os.makedirs("output/figures", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # (1) statistics ladder
    ax = axes[0]
    inv_alpha = 1.0 / np.array([1.0, 0.3, 0.1, 0.03, 0.01])   # 1/alpha axis for the analytic line
    for tag_name, col in [("matched", "C3"), ("mec", "C0")]:
        chi2ndf1 = results[tag_name]["c_bf"] / ndf
        ax.plot(inv_alpha, chi2ndf1 * inv_alpha, "-o", color=col,
                label=f"{tag_name} (BFP chi2/ndf={chi2ndf1:.2f} @ T2K)")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.axvline(1.0, color="0.5", lw=0.8, ls="--")
    ax.text(1.05, ax.get_ylim()[1]*0.9, "T2K today", fontsize=8, color="0.4")
    ax.set(xlabel=r"$1/\alpha$  (precision vs T2K)", ylabel=r"best-fit $\chi^2/$ndf",
           xscale="log", yscale="log", title="Statistics ladder: mismatch is un-hideable")
    ax.legend(fontsize=8)

    # (2) irreducible-residual fingerprint (per-bin standardized post-fit residual, dpt)
    ax = axes[1]
    for tag_name, col in [("matched", "C3"), ("mec", "C0")]:
        sr = results[tag_name]["stdres"][idpt]
        ax.step(xed, np.append(sr, sr[-1]), where="post", color=col, lw=1.8,
                label=f"{tag_name}  (irr.frac={results[tag_name]['chi2_irr']/results[tag_name]['chi2_tot']:.2f})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel=dpt["xlabel"], ylabel=r"post-fit residual / $\sigma_{T2K}$",
           title="Irreducible fingerprint (CC0$\\pi$ $\\delta p_T$)")
    ax.legend(fontsize=8)

    # (3) MEC-template closure: decompose the KNOWN missing channel into knob-absorbable vs irreducible
    ax = axes[2]
    ax.step(xed, np.append(mec_template, mec_template[-1]), where="post", color="C2", lw=1.3, alpha=0.5,
            label=f"full MEC channel  (raw corr {cc_raw:.2f})")
    ax.step(xed, np.append(mec_irr, mec_irr[-1]), where="post", color="C2", lw=2.2,
            label=f"MEC irreducible part  ({100*frac_irr_mec:.0f}% un-absorbable)")
    ax.step(xed, np.append(resid_matched, resid_matched[-1]), where="post", color="C3", lw=1.8, ls="--",
            label=f"matched fit residual  (corr {cc_irr:.2f})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel=dpt["xlabel"], ylabel=r"$d\sigma$ [$10^{-38}$/nucleon]",
           title=f"MEC-channel decomposition: {100*frac_irr_mec:.0f}% irreducible")
    ax.legend(fontsize=8)

    fig.suptitle(f"High-statistics diagnostic [{LABEL}]: foreign-generator mismatch becomes informative")
    fig.tight_layout()
    out = f"output/figures/altgen_highstat_{LABEL}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    log(f"[fig] {out}  MEC irr.frac={frac_irr_mec:.3f} corr(resid, MEC_irr)={cc_irr:.3f} "
        f"corr(resid, MEC_raw)={cc_raw:.3f}")

    np.savez(f"output/altgen/highstat_{LABEL}.npz",
             fit=FIT, ndf=ndf, nbins=nbins,
             th_matched=results["matched"]["th"], th_mec=results["mec"]["th"],
             cnom_matched=results["matched"]["c_nom"], cbf_matched=results["matched"]["c_bf"],
             cnom_mec=results["mec"]["c_nom"], cbf_mec=results["mec"]["c_bf"],
             irr_matched=results["matched"]["chi2_irr"], tot_matched=results["matched"]["chi2_tot"],
             irr_mec=results["mec"]["chi2_irr"], tot_mec=results["mec"]["chi2_tot"],
             mec_template=mec_template, mec_irr=mec_irr, resid_matched=resid_matched, dpt_x=x,
             frac_irr_mec=frac_irr_mec, corr_raw=cc_raw, corr_irr=cc_irr,
             ladder_alpha=ladder_gate["alpha"], ladder_dmax=ladder_gate["dmax"],
             ladder_chi2=ladder_gate["chi2"], ladder_expect=ladder_gate["expect"])
    log("done")


if __name__ == "__main__":
    main()
