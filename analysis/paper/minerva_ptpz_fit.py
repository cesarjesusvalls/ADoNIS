"""MINERvA CC0pi / QE-like muon-kinematics Gate-I gradient driver (p_T^mu, p_||^mu, 2D p_T-p_||).

Produces output/altgen/physfit_minerva_ptpz.npz in the physfit schema so the Jacobian stacks into the
multi-sample gradient figure.  Same engine as physical_fit.py / minerva_fit.py; the SAMPLE is the
MINERvA "qelike" muon double-differential (Ruterbories et al., PRD 99 (2019) 012004, arXiv:1811.12569):

  * bank      : output/paper_banks_p4/nu_MINERvA_C (weak probe, MINERvA flux, carbon)
  * signal    : NUISANCE SignalDef::isCC0pi_MINERvAPTPZ -- CC, theta_mu < 20 deg, exactly one muon,
                ZERO final-state mesons (and no heavy baryons / pi0 / hard photons), p_mu >= 1.5 GeV.
                INCLUSIVE over nucleons -- NO proton requirement (unlike the CC0pi-Np STV in
                minerva_fit.py).  This keeps the muon-vertex information the STV variables integrate out.
  * observables: p_T^mu (transverse), p_||^mu (longitudinal), and the 2D p_T-p_|| double-differential.

The 28-knob order/priors/SYST/N_BINS are IMPORTED from physical_fit so rows align 1:1.  Bin edges are
design_edges (per axis); the overall per-bin scale cancels in J/sigma (Fisher/Gate-I are unaffected).

    srun ... $ADONIS_PY -u -m analysis.paper.minerva_ptpz_fit
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.reweight import bank_plot as BP, bank_reweight as BR
from adonis.reweight.reweight_model import nominal_knobs
from adonis.constants import PDG_MESONS
from analysis.paper import info_content as IC
from analysis.paper.physical_fit import (
    knobs_of, theta_nominal, PRIOR, PNAMES, NPAR, SYST, N_BINS, design_edges,
)
from analysis.paper.t2k_pcos_fit import bin2d_dataset

BANKDIR = os.environ.get("ADONIS_MINERVA_BANK", "output/paper_banks_p4/nu_MINERvA_C/merged")
LABEL = os.environ.get("ADONIS_LABEL", "physfit_minerva_ptpz")
COS20 = float(np.cos(np.deg2rad(20.0)))     # theta_mu < 20 deg
MU_LO = 1500.0                              # p_mu >= 1.5 GeV (ME lower cut)
N_PT = int(os.environ.get("ADONIS_PTPZ_NPT", "7"))     # 2D p_T axis bins
N_PZ = int(os.environ.get("ADONIS_PTPZ_NPZ", "7"))     # 2D p_|| axis bins
CONV0 = 1e-33 / 12.0 * 1e38                             # per-nucleon 1e-38 (carbon); cancels in J/sigma
_MESONS = list(PDG_MESONS)


def bin1d_dataset(name, key, mask, vals, edges, conv):
    """A 1D physfit dataset (design edges, overflow fold), mirroring minerva_fit.build_minerva_datasets."""
    nb = len(edges) - 1; bw = np.diff(edges)
    eps = (edges[-1] - edges[0]) * 1e-12
    vc = np.clip(vals, edges[0] + eps, edges[-1] - eps)
    sel, bidx, nbA = IC._bin(mask, vc, edges); assert nbA == nb
    return dict(name=name, key=key, sel_idx=sel, binidx=bidx, nbin=nb,
                scale_bin=conv / bw, offset=np.zeros(nb), edges=edges)


def _fill(d, w0):
    central = IC.bin_w(d, w0)
    sw2 = np.bincount(d["binidx"], weights=np.asarray(w0)[d["sel_idx"]] ** 2, minlength=d["nbin"])
    mcerr = d["scale_bin"] * np.sqrt(sw2)
    var = (SYST * central) ** 2 + mcerr ** 2
    # empty bins (esp. in the 2D grid) carry no info; sigma=inf so 0/0 does not poison the Fisher
    d.update(data=central, sigma=np.where(var > 0, np.sqrt(var), np.inf), mcerr=mcerr)
    return d


def build_ptpz_datasets(B, w0, log):
    """p_T^mu (1D), p_||^mu (1D), 2D p_T-p_|| over the qelike (0-meson, theta_mu<20, no-proton) selection."""
    kmu = B["k_lep"].astype(np.float64)
    pmu = np.linalg.norm(kmu[:, 1:], axis=1)
    cmu = kmu[:, 3] / np.maximum(pmu, 1e-9)
    pt = np.sqrt(kmu[:, 1] ** 2 + kmu[:, 2] ** 2)                  # transverse muon momentum
    pz = kmu[:, 3]                                                 # longitudinal muon momentum
    n_meson = BP._event_sum(B, np.isin(B["fs_pid"], _MESONS).astype(float))
    mask = (n_meson == 0) & (cmu > COS20) & (pmu >= MU_LO)         # isCC0pi_MINERvAPTPZ (hadron-inclusive)
    nsel = int(mask.sum())
    log(f"  MINERvA qelike (0-meson, theta_mu<20, p_mu>=1.5GeV, NO proton): {nsel}/{len(w0)} ({nsel/len(w0):.2%})")
    ex = design_edges(pt[mask], N_PT, domain=None)
    ey = design_edges(pz[mask], N_PZ, domain=None)
    ds = [
        _fill(bin1d_dataset("MINERvA p_T^mu", "mnv_ptmu", mask, pt, design_edges(pt[mask], N_BINS), CONV0 * 1000.0), w0),
        _fill(bin1d_dataset("MINERvA p_||^mu", "mnv_pzmu", mask, pz, design_edges(pz[mask], N_BINS), CONV0 * 1000.0), w0),
        _fill(bin2d_dataset("MINERvA 2D p_T-p_||", "mnv_ptpl", mask, pt, pz, ex, ey, CONV0), w0),
    ]
    for d in ds:
        log(f"  [{d['name']}] {d['nbin']} bins | MC err med "
            f"{np.median(d['mcerr']/np.maximum(d['data'],1e-30)):.2%}")
    return ds


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    B = BP.load_bank(BANKDIR); JB = BR.to_jax(B); grids = BR.default_grids(); nom = nominal_knobs()
    w0 = np.asarray(BR.weight_jit(JB, nom, grids))
    log(f"bank {BANKDIR}: {len(w0)} events | {NPAR} knobs | syst {SYST:.0%}")
    ds = build_ptpz_datasets(B, w0, log)
    nbins = sum(d["nbin"] for d in ds)
    log(f"{len(ds)} datasets, {nbins} bins")

    th0 = theta_nominal(nom)
    def wf(theta, JB):
        return BR.bank_weight(JB, knobs_of(theta, nom), grids)
    wf_jit = jax.jit(wf)
    jvp_wf = jax.jit(lambda th, tang, JB: jax.jvp(lambda t: wf(t, JB), (th,), (tang,))[1])

    w0b = np.asarray(wf_jit(jnp.asarray(th0), JB))
    log(f"(V1) nominal identity |w(th0)-w_nominal|_max = {np.max(np.abs(w0b-w0)):.2e}")

    J = np.zeros((nbins, NPAR))
    row0 = np.cumsum([0] + [d["nbin"] for d in ds])
    for k in range(NPAR):
        g = np.asarray(jvp_wf(jnp.asarray(th0), jnp.zeros(NPAR).at[k].set(1.0), JB))
        for j, d in enumerate(ds):
            J[row0[j]:row0[j+1], k] = IC.bin_w0(d, g)
        log(f"  jvp {k+1:2d}/{NPAR} {PNAMES[k]}")
    sigma = np.concatenate([d["sigma"] for d in ds])

    Jw = J / sigma[:, None]
    F = Jw.T @ Jw
    V = np.linalg.inv(F + np.diag(1.0 / PRIOR ** 2))
    sig_post = np.sqrt(np.diag(V)); shrink = sig_post / PRIOR
    reach = np.sqrt(np.maximum(np.diag(F), 0.0))
    order = np.argsort(shrink)
    print(f"\n==== MINERvA qelike (p_T,p_||) GATE I (Asimov Fisher, {nbins} bins, syst {SYST:.0%}) ====")
    print(f"{'knob':>18} {'prior':>7} {'sqrtFkk':>10} {'shrink':>7}")
    for k in order:
        print(f"{PNAMES[k]:>18} {PRIOR[k]:7.2f} {reach[k]:10.3g} {shrink[k]:7.2f}")
    print(f"\n{int((shrink<0.5).sum())}/{NPAR} knobs pass Gate I on MINERvA qelike alone")
    print("\n---- verification: weak probe -> axial AND vector must be nonzero ----")
    for name in ["M_A_qe", "axial_strength", "vector_strength", "kF_sf", "sabs", "qe_norm"]:
        k = PNAMES.index(name); print(f"    {name:>18}  sqrtF={reach[k]:.4g}   {'OK' if reach[k] > 0 else 'ZERO!!'}")

    os.makedirs("output/altgen", exist_ok=True)
    outpath = f"output/altgen/{LABEL}.npz"
    np.savez(outpath, J=J, sigma=sigma, F=F, V=V, prior=PRIOR, sig_post=sig_post, shrink=shrink,
             pnames=PNAMES, nbins=nbins, row0=row0,
             dsnames=[d["name"] for d in ds], dskeys=[d["key"] for d in ds],
             **{f"{d['key']}_edges": d["edges"] for d in ds if d["key"] in ("mnv_ptmu", "mnv_pzmu")})
    log(f"[out] {outpath}")
    log("done")


if __name__ == "__main__":
    main()
