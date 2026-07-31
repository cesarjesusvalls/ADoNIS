"""T2K CC0pi 2D d2sigma/dp_mu dcostheta_mu Gate-I gradient driver (the flagship lepton-kinematics sample).

Produces output/altgen/physfit_t2k_pcos.npz in the SAME physfit schema as physfit_gate1.npz so the
Jacobian stacks into the multi-sample gradient figure.  Engine (bank -> jvp per knob -> Fisher) is
identical to physical_fit.py / minerva_fit.py; only the SAMPLE + observable differ:

  * bank      : output/paper_banks_p4/nu_T2K_C (weak probe, T2K flux, carbon), 4.69M events
  * signal    : T2K CC0pi, "Analysis I" phase space = NUISANCE SignalDef::isT2K_CC0pi(kAnalysis_I):
                a numu CC event with ZERO final-state mesons -- INCLUSIVE over hadrons (NO proton
                requirement, NO muon phase-space cut).  This is deliberately looser than the CC0pi-Np
                STV selection (physfit_gate1): it keeps the muon-vertex information that the transverse
                variables integrate away.  Ref: Abe et al., PRD 93 (2016) 112012, arXiv:1602.03652.
  * observable: the 2D double-differential in (p_mu, cos theta_mu), flattened to n_pmu*n_cos bins.

The 28-knob order/priors/SYST/N_BINS are IMPORTED from physical_fit so the Jacobian rows align 1:1.
Bin edges are design_edges per axis (published 2D edges are irregular; the gradient figure uses the
5%-Asimov model, so uniform design bins are the established choice -- LOGGED as such).  The overall
per-bin scale cancels in J/sigma, so it does not affect the Fisher/Gate-I -- only display units.

    srun ... $ADONIS_PY -u -m analysis.paper.t2k_pcos_fit
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

BANKDIR = os.environ.get("ADONIS_T2K_BANK", "output/paper_banks_p4/nu_T2K_C/merged")
LABEL = os.environ.get("ADONIS_LABEL", "physfit_t2k_pcos")
N_PMU = int(os.environ.get("ADONIS_PCOS_NPMU", "8"))    # p_mu axis bins
N_COS = int(os.environ.get("ADONIS_PCOS_NCOS", "6"))    # cos(theta_mu) axis bins
CONV0 = 1e-33 / 12.0 * 1e38                              # per-nucleon 1e-38 (carbon); cancels in J/sigma
_MESONS = list(PDG_MESONS)                               # np.isin needs a list -- a frozenset matches NOTHING


def bin2d_dataset(name, key, mask, vx, vy, ex, ey, conv):
    """A flattened 2D histogram as a physfit dataset: bin b = iy*nx + ix.  scale_bin = conv / bin-area.
    Overflow folds into the edge bins (clip).  Returns the dataset dict (data/sigma filled by caller)."""
    nx, ny = len(ex) - 1, len(ey) - 1
    nb = nx * ny
    epsx = (ex[-1] - ex[0]) * 1e-12; epsy = (ey[-1] - ey[0]) * 1e-12
    vxc = np.clip(vx, ex[0] + epsx, ex[-1] - epsx)
    vyc = np.clip(vy, ey[0] + epsy, ey[-1] - epsy)
    ix = np.clip(np.searchsorted(ex, vxc) - 1, 0, nx - 1)
    iy = np.clip(np.searchsorted(ey, vyc) - 1, 0, ny - 1)
    comb = iy * nx + ix
    sel = np.where(mask)[0].astype(np.int64)
    binidx = comb[sel].astype(np.int64)
    area = (np.diff(ey)[:, None] * np.diff(ex)[None, :]).ravel()   # index iy*nx+ix -> dy[iy]*dx[ix]
    return dict(name=name, key=key, sel_idx=sel, binidx=binidx, nbin=nb,
                scale_bin=conv / area, offset=np.zeros(nb), edges=np.arange(nb + 1, dtype=float),
                ex=ex, ey=ey, nx=nx, ny=ny)


def build_t2k_pcos_datasets(B, w0, log):
    """One 2D observable: (p_mu, cos theta_mu) over the CC0pi Analysis-I (0-meson) selection."""
    kmu = B["k_lep"].astype(np.float64)
    pmu = np.linalg.norm(kmu[:, 1:], axis=1)
    cmu = kmu[:, 3] / np.maximum(pmu, 1e-9)
    n_meson = BP._event_sum(B, np.isin(B["fs_pid"], _MESONS).astype(float))
    mask = (n_meson == 0)                                          # CC0pi, inclusive over hadrons
    nsel = int(mask.sum())
    log(f"  T2K CC0pi (Analysis I, 0-meson, hadron-inclusive): {nsel}/{len(w0)} events ({nsel/len(w0):.2%})")
    ex = design_edges(pmu[mask], N_PMU, domain=None)               # p_mu [min, p99] + fold
    ey = design_edges(cmu[mask], N_COS, domain=(float(np.min(cmu[mask])), 1.0))   # cos bounded above at 1
    d = bin2d_dataset("T2K CC0pi 2D pmu-cosmu", "t2k_pcos", mask, pmu, cmu, ex, ey, CONV0)
    central = IC.bin_w(d, w0)
    sw2 = np.bincount(d["binidx"], weights=np.asarray(w0)[d["sel_idx"]] ** 2, minlength=d["nbin"])
    mcerr = d["scale_bin"] * np.sqrt(sw2)
    var = (SYST * central) ** 2 + mcerr ** 2
    # EMPTY 2D cells (no events -> central 0, mcerr 0) carry no information; give them sigma=inf (zero
    # weight in J/sigma, J is already 0 there) so 0/0 does not poison the Fisher -- same as beam_fisher.
    sigma = np.where(var > 0, np.sqrt(var), np.inf)
    d.update(data=central, sigma=sigma, mcerr=mcerr)
    log(f"  [t2k_pcos] {d['nbin']} bins ({N_PMU} p_mu x {N_COS} cos) | p_mu[{ex[0]:.0f},{ex[-1]:.0f}]MeV "
        f"cos[{ey[0]:.3f},{ey[-1]:.3f}] | MC err med {np.median(mcerr/np.maximum(central,1e-30)):.2%}")
    return [d]


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    B = BP.load_bank(BANKDIR); JB = BR.to_jax(B); grids = BR.default_grids(); nom = nominal_knobs()
    w0 = np.asarray(BR.weight_jit(JB, nom, grids))
    log(f"bank {BANKDIR}: {len(w0)} events | {NPAR} knobs | 2D {N_PMU}x{N_COS} | syst {SYST:.0%}")
    ds = build_t2k_pcos_datasets(B, w0, log)
    nbins = sum(d["nbin"] for d in ds)

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
    print(f"\n==== T2K CC0pi 2D (p_mu,cos) GATE I (Asimov Fisher, {nbins} bins, syst {SYST:.0%}) ====")
    print(f"{'knob':>18} {'prior':>7} {'sqrtFkk':>10} {'shrink':>7}")
    for k in order:
        print(f"{PNAMES[k]:>18} {PRIOR[k]:7.2f} {reach[k]:10.3g} {shrink[k]:7.2f}")
    print(f"\n{int((shrink<0.5).sum())}/{NPAR} knobs pass Gate I on T2K CC0pi 2D alone")
    print("\n---- verification: weak probe -> axial AND vector must be nonzero ----")
    for name in ["M_A_qe", "axial_strength", "vector_strength", "kF_sf", "sabs", "qe_norm"]:
        k = PNAMES.index(name); print(f"    {name:>18}  sqrtF={reach[k]:.4g}   {'OK' if reach[k] > 0 else 'ZERO!!'}")

    os.makedirs("output/altgen", exist_ok=True)
    outpath = f"output/altgen/{LABEL}.npz"
    np.savez(outpath, J=J, sigma=sigma, F=F, V=V, prior=PRIOR, sig_post=sig_post, shrink=shrink,
             pnames=PNAMES, nbins=nbins, row0=row0,
             dsnames=[d["name"] for d in ds], dskeys=[d["key"] for d in ds])
    log(f"[out] {outpath}")
    log("done")


if __name__ == "__main__":
    main()
