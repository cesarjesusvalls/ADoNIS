"""JOINT CC0pi+CC1pi fit of the ADoNIS differentiable bank to the GENIE AR23 fake datasets.

The single-observable dpt fit (fit_fakedata.py, logbook 9) railed the RES sector (sabs, res_norm)
as a shape compensation even though the RES->CC0pi RATES agree (9b). The llh_surface study showed
CC1pi+Np data tightens res_norm 3.2x on real T2K data. This script tests the same cure on the
foreign-generator fake data: 5 datasets (CC0pi dpt+dat [1e-38/nucleon], CC1pi pN+dpTT+daT [nb/CH]),
full T2K covariances as the error model, GENIE central values.

REUSE (single source of truth): info_content.build_datasets assembles selections/bins/units/free-H
offsets for the MODEL side; we only swap each dataset's central values for the GENIE ones
(CC0pi: GENIE dsig; CC1pi: GENIE-C dsig + the SAME frozen free-H offset used in the model, so the
H contribution cancels exactly in the residual -- only the C part is foreign).

Fit: Gauss-Newton/LM on the exact bank reweight; Jacobian per iteration via jax.jvp one knob at a
time (info_content pattern; JB threaded as an argument so it is never a jit constant -> no OOM).
Knobs (6, matched to fit_fakedata.py): M_A_qe, qe_norm, kF_sf, s_NN_el(common), sabs, res_norm.
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
FAKE = os.environ.get("ADONIS_FAKEDATA", "output/altgen/fakedata_ccqeres.npz")
TAG = os.environ.get("ADONIS_FAKE_TAG", "")          # npz key suffix ("", "_full")
LABEL = os.environ.get("ADONIS_FIT_LABEL", "joint")  # output filename label

FIT = ["M_A_qe", "qe_norm", "kF_sf", "s_NN_el", "sabs", "res_norm"]
LO = np.array([0.5, 0.3, 0.6, 0.3, 0.3, 0.3])
HI = np.array([2.0, 3.0, 1.6, 3.0, 3.0, 3.0])


def knobs_from(theta, nom):
    upd = {}
    for i, key in enumerate(FIT):
        if key == "s_NN_el":
            upd["s_NN_elastic"] = theta[i] * jnp.ones(3)
        else:
            upd[key] = theta[i]
    return nom._replace(**upd)


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

    B = BP.load_bank(BANKDIR); JB = BR.to_jax(B); grids = BR.default_grids(); nom = nominal_knobs()
    log(f"bank {len(B['w0'])} events")
    ds = IC.build_datasets(B)
    log(f"{len(ds)} datasets assembled (model side; selections/bins/offsets from analysis.paper.info_content)")

    # ---- swap central values -> GENIE fake data ------------------------------------------------- #
    d0 = np.load(FAKE, allow_pickle=True)
    for d in ds:
        if d["channel"] == "CC0pi":
            d["data"] = np.asarray(d0[f"{d['key']}{TAG}_dsig"])               # 1e-38 cm^2/unit/nucleon
        else:
            d["data"] = np.asarray(d0[f"cc1pi_{d['key']}{TAG}_dsig_C"]) + d["offset"]  # gen-C + frozen H
    log(f"targets swapped to fake data {FAKE} (tag='{TAG}', label={LABEL})")

    # ---- differentiable weights: JB as ARGUMENT (info_content pattern) -------------------------- #
    def wf(theta, JB):
        return BR.bank_weight(JB, knobs_from(theta, nom), grids)
    wf_jit = jax.jit(wf)
    jvp_wf = jax.jit(lambda th, tang, JB: jax.jvp(lambda t: wf(t, JB), (th,), (tang,))[1])
    NP = len(FIT)

    def model_bins(theta_np):
        w = np.asarray(wf_jit(jnp.asarray(theta_np), JB))
        return [IC.bin_w(d, w) for d in ds]

    def jac_bins(theta_np):
        Js = [np.zeros((d["nbin"], NP)) for d in ds]
        for k in range(NP):
            g = np.asarray(jvp_wf(jnp.asarray(theta_np), jnp.zeros(NP).at[k].set(1.0), JB))
            for j, d in enumerate(ds):
                Js[j][:, k] = IC.bin_w0(d, g)
        return Js

    def chi2_of(mods):
        tot, per = 0.0, []
        for d, m in zip(ds, mods):
            r = m - d["data"]; c = float(r @ d["Cinv"] @ r); per.append(c); tot += c
        return tot, per

    th = np.array([1.0] * NP)
    mods = model_bins(th)
    c_nom, per_nom = chi2_of(mods)
    nbins = sum(d["nbin"] for d in ds)
    log(f"NOMINAL joint chi2 = {c_nom:.1f}  (ndf={nbins}, chi2/ndf={c_nom/nbins:.2f})")
    for d, c in zip(ds, per_nom):
        log(f"    {d['name']:12s} chi2/nbin = {c/d['nbin']:.2f}")

    # ---- Levenberg-Marquardt --------------------------------------------------------------------- #
    lam = 1e-3; c_cur = c_nom
    NIT = int(os.environ.get("ALTGEN_NIT", "40"))
    for it in range(NIT):
        Js = jac_bins(th)
        A = np.zeros((NP, NP)); g = np.zeros(NP)
        for d, m, J in zip(ds, mods, Js):
            r = m - d["data"]
            A += J.T @ d["Cinv"] @ J; g += J.T @ d["Cinv"] @ r
        for _ in range(12):
            dth = np.linalg.solve(A + lam * np.diag(np.maximum(np.diag(A), 1e-12)), -g)
            th_try = np.clip(th + dth, LO, HI)
            mods_try = model_bins(th_try)
            c_try, _ = chi2_of(mods_try)
            if c_try < c_cur:
                th, mods, c_cur = th_try, mods_try, c_try; lam = max(lam / 3, 1e-8); break
            lam *= 5
        log(f"  it {it:2d}  chi2={c_cur:8.2f}  " + " ".join(f"{k}={th[i]:.3f}" for i, k in enumerate(FIT)))
        if np.linalg.norm(dth) < 1e-5:
            log("  converged (step < 1e-5)"); break

    # ---- errors: Gauss-Newton Hessian (pinv; rails invalid) -------------------------------------- #
    Js = jac_bins(th); A = np.zeros((NP, NP))
    for d, J in zip(ds, Js):
        A += J.T @ d["Cinv"] @ J
    V = np.linalg.pinv(A, rcond=1e-12); sig = np.sqrt(np.abs(np.diag(V)))
    railed = [(abs(th[i] - LO[i]) < 1e-3 or abs(th[i] - HI[i]) < 1e-3) for i in range(NP)]
    _, per_bf = chi2_of(mods)
    ndf = nbins - NP
    print(f"\n==== JOINT CC0pi+CC1pi fit [{LABEL}] to {FAKE} (tag='{TAG}') "
          f"(chi2 {c_nom:.1f} -> {c_cur:.1f}; ndf {nbins} -> {ndf}) ====")
    print(f"{'knob':>20} {'BFP':>8} {'+/-':>8} {'pull':>7} {'flag':>6}")
    for i, k in enumerate(FIT):
        print(f"{k:>20} {th[i]:8.3f} {sig[i]:8.3f} {(th[i]-1.0)/sig[i] if sig[i]>0 else 0:7.2f} "
              f"{'RAIL' if railed[i] else '':>6}")
    print("\nper-dataset chi2/nbin (nominal -> BFP):")
    for d, cn, cb in zip(ds, per_nom, per_bf):
        print(f"    {d['name']:12s} {cn/d['nbin']:6.2f} -> {cb/d['nbin']:6.2f}")

    # ---- figure: 5 panels + pulls ---------------------------------------------------------------- #
    mods_nom = model_bins(np.ones(NP))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, d, mn, mb in zip(axes.flat, ds, mods_nom, mods):
        ctr = 0.5 * (d["edges"][1:] + d["edges"][:-1]) / d["xscale"]
        xed = d["edges"] / d["xscale"]
        ax.errorbar(ctr, d["data"], yerr=d["sigma"], fmt="o", color="k", ms=4, capsize=2, label=f"{LABEL} fake data")
        ax.step(xed, np.append(mn, mn[-1]), where="post", color="0.5", ls=":", label="ADoNIS nominal")
        ax.step(xed, np.append(mb, mb[-1]), where="post", color="C0", lw=1.8, label="ADoNIS joint BFP")
        ax.set(xlabel=d["xlabel"], title=d["name"]); ax.set_ylim(bottom=0)
        if d is ds[0]:
            ax.legend(fontsize=8)
    axp = axes.flat[5]
    pulls = [(th[i] - 1.0) / sig[i] if sig[i] > 0 else 0.0 for i in range(NP)]
    cols = ["C3" if railed[i] else ("C1" if abs(p) > 2 else "C0") for i, p in enumerate(pulls)]
    axp.barh(FIT[::-1], pulls[::-1], color=cols[::-1])
    axp.axvline(0, color="k", lw=0.8); axp.axvline(2, ls="--", color="0.6"); axp.axvline(-2, ls="--", color="0.6")
    axp.set(xlabel="pull (BFP-1)/sigma", title="joint pulls (red=RAIL)")
    for i, k in enumerate(FIT[::-1]):
        j = NP - 1 - i
        axp.text(pulls[j], i, f"  {th[j]:.2f}", va="center", fontsize=8)
    fig.suptitle(f"JOINT CC0pi+CC1pi ADoNIS fit ({LABEL})  "
                 f"(chi2/ndf {c_nom/nbins:.2f} -> {c_cur/ndf:.2f})")
    fig.tight_layout()
    os.makedirs("output/figures", exist_ok=True)
    out = f"output/figures/altgen_fit_{LABEL}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    np.savez(f"output/altgen/fit_{LABEL}.npz", fit=FIT, bfp=th, sig=sig, V=V, chi2_nom=c_nom,
             chi2_bf=c_cur, nbins=nbins, ndf=ndf, per_nom=per_nom, per_bf=per_bf,
             names=[d["name"] for d in ds])
    log(f"[fig] {out}")


if __name__ == "__main__":
    main()
