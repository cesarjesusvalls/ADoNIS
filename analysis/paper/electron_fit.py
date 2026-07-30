"""(e,e') electron-scattering Gate-I gradient driver (EM probe).

Produces output/altgen/physfit_electron.npz in the SAME schema as physfit_gate1.npz so the Jacobian
can be stacked into the multi-sample gradient figure.  The engine (bank -> jvp per knob -> Fisher) is
identical to analysis/paper/physical_fit.py / minerva_fit.py; the sample is the (e,e') beam:

  * bank    : output/beam_e_C_hv (monochromatic e- beam on 12C, E=2222 MeV, theta_e in [14,17] deg,
              EM hard vertex, QE+RES; carries the hv_ vector/FF/SF records added for EM gradients)
  * signal  : all accepted (in-acceptance) events, split by channel into QE and RES
  * observable : omega = energy transfer [MeV], one spectrum per channel (e_qe, e_res)

Because the probe is a PHOTON there is no axial current: the EM hv_ records collapse the QE axial and
all RES hard-vertex records to identity, so d(sigma)/d{M_A_qe,axial_strength,res_axial_strength} and the
EM RES vector/FF knobs are EXACTLY zero -- the electron probe's correctness gate (verified below).  The
vector/Sachs-FF, spectral-function and FSI knobs carry real gradients.

    srun ... $ADONIS_PY -u -m analysis.paper.electron_fit
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
from analysis.paper import info_content as IC
# align the 28-knob basis EXACTLY with physfit_gate1 (same order, priors, SYST, N_BINS, binning helpers)
from analysis.paper.physical_fit import (
    knobs_of, theta_nominal, PRIOR, PNAMES, NPAR, SYST, N_BINS, design_edges,
)

BANKDIR = os.environ.get("ADONIS_ELECTRON_BANK", "output/beam_e_C_hv/merged")
LABEL = os.environ.get("ADONIS_LABEL", "physfit_electron")

# omega [MeV] per-nucleon 1e-38 display conversion (carbon); common to J/central/mcerr in every bin so it
# CANCELS in J/sigma and does not affect the Fisher/Gate-I -- it only sets display units of central/sigma.
CONV0 = 1e-33 / 12.0 * 1e38
# axial knobs that MUST be exactly zero for an EM probe (photon has no axial current); the EM RES
# hard-vertex records are identity for now (delta_strength EM is a v2 item) -> also zero here.
_EM_ZERO = ["M_A_qe", "axial_strength", "res_axial_strength", "M_A_res", "pion_pole", "delta_strength"]


def build_electron_datasets(B, w0, log):
    """2 observables: omega for QE (channel==0) and RES (channel==1). design_edges over [min,p99]+fold,
    diagonal syst+MC error, Asimov centrals.  Signal = in-acceptance events with non-zero weight."""
    omega = np.asarray(B["omega"], float)
    chan = np.asarray(B["channel"]).astype(int)
    good = np.asarray(w0) > 0                                   # drop zero-weight padding rows
    obs_defs = [("(e,e') QE omega",  "e_qe",  chan == 0),
                ("(e,e') RES omega", "e_res", chan == 1)]
    ds = []
    for name, dkey, chmask in obs_defs:
        mask = good & chmask
        v = omega[mask]
        edges = design_edges(v, N_BINS, domain=None)           # [min, p99] + overflow fold
        nb = len(edges) - 1; bw = np.diff(edges)
        eps = (edges[-1] - edges[0]) * 1e-12
        vc = np.clip(omega, edges[0] + eps, edges[-1] - eps)
        sel, bidx, nbA = IC._bin(mask, vc, edges); assert nbA == nb
        d = dict(name=name, key=dkey, sel_idx=sel, binidx=bidx, nbin=nb,
                 scale_bin=CONV0 / bw, offset=np.zeros(nb), edges=edges)
        central = IC.bin_w(d, w0)
        sw2 = np.bincount(bidx, weights=np.asarray(w0)[sel] ** 2, minlength=nb)
        mcerr = d["scale_bin"] * np.sqrt(sw2)
        var = (SYST * central) ** 2 + mcerr ** 2
        d.update(data=central, sigma=np.sqrt(var), mcerr=mcerr)
        ds.append(d)
        log(f"  [{name}] {int(mask.sum())} evt | {nb} bins over [{edges[0]:.3g},{edges[-1]:.3g}] MeV "
            f"| MC err med {np.median(mcerr/np.maximum(central,1e-30)):.2%}")
    return ds


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    B = BP.load_bank(BANKDIR); JB = BR.to_jax(B); grids = BR.default_grids(); nom = nominal_knobs()
    w0 = np.asarray(BR.weight_jit(JB, nom, grids))
    log(f"bank {BANKDIR}: {len(w0)} events (EM probe) | {NPAR} knobs | {N_BINS} bins/obs | syst {SYST:.0%}")
    ds = build_electron_datasets(B, w0, log)
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
    print(f"\n==== (e,e') GATE I (Asimov Fisher, {nbins} bins, syst {SYST:.0%}) ====")
    print(f"{'knob':>18} {'prior':>7} {'sqrtFkk':>10} {'shrink':>7}")
    for k in order:
        print(f"{PNAMES[k]:>18} {PRIOR[k]:7.2f} {reach[k]:10.3g} {shrink[k]:7.2f}")
    print(f"\n{int((shrink<0.5).sum())}/{NPAR} knobs pass Gate I on (e,e') alone")

    # CORRECTNESS GATE: an EM probe has NO axial current -> these knobs' reach must be EXACTLY zero,
    # while the vector/SF/FSI knobs must be nonzero.
    print("\n---- EM correctness gate: axial/RES-hard-vertex knobs must be ZERO ----")
    bad = []
    for name in _EM_ZERO:
        r = reach[PNAMES.index(name)]
        ok = r < 1e-9
        print(f"    {name:>20}  sqrtF={r:.3e}   {'ZERO ok' if ok else 'NONZERO!!'}")
        if not ok: bad.append(name)
    print("  vector/SF/FSI (must be > 0):")
    for name in ["vector_strength", "mu_p", "gep", "kF_sf", "sf_norm", "sabs", "s_NN_elastic[0]"]:
        r = reach[PNAMES.index(name)]
        print(f"    {name:>20}  sqrtF={r:.4g}   {'OK' if r > 0 else 'ZERO!!'}")
    if bad:
        raise SystemExit(f"EM correctness gate FAILED: nonzero axial gradient in {bad}")

    os.makedirs("output/altgen", exist_ok=True)
    outpath = f"output/altgen/{LABEL}.npz"
    np.savez(outpath, J=J, sigma=sigma, F=F, V=V, prior=PRIOR, sig_post=sig_post, shrink=shrink,
             pnames=PNAMES, nbins=nbins, row0=row0,
             dsnames=[d["name"] for d in ds], dskeys=[d["key"] for d in ds],
             **{f"{d['key']}_edges": d["edges"] for d in ds},
             **{f"{d['key']}_central": d["data"] for d in ds})
    log(f"[out] {outpath}")
    log("done")


if __name__ == "__main__":
    main()
