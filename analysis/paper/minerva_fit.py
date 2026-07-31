"""MINERvA CC0pi-Np Gate-I gradient driver.

Produces output/altgen/physfit_minerva.npz in the SAME schema as physfit_gate1.npz so the
Jacobian can be stacked into the multi-sample gradient figure.  The engine (bank -> jvp per knob
-> Fisher) is identical to analysis/paper/physical_fit.py; only the SAMPLE differs:

  * bank    : output/paper_banks_p4/nu_MINERvA_C (weak probe, MINERvA flux, carbon)
  * signal  : MINERvA CC0pi-Np (Cai et al., MINERvA:2018hba / NUISANCE MINERvA_CC0pinp)
              - muon  : 1.5 <= p_mu <= 10 GeV  (1500-10000 MeV), theta_mu < 20 deg (cos > cos20)
              - lead p: 0.45 <= p_p <= 1.2 GeV/c (450-1200 MeV), theta_p < 70 deg (cos > cos70)
              - CC0pi : zero final-state mesons; >= 1 proton in the acceptance (global-leading, ge1)
  * observables : dat [deg] (0-180), pn [MeV] (p_n^recon), dpt [MeV]

The 28-knob order, priors, SYST and N_BINS are IMPORTED from physical_fit so the Jacobian rows
align 1:1 with physfit_gate1.  Bin edges: no NUISANCE STV CC0pi-Np data files exist under
../nuisance (only the unrelated MINERvA/CC0pi_3D triple-differential set), so edges are built with
physical_fit.design_edges on the selected events (LOGGED as such).  The gradient-figure sigma is the
5%-Asimov model (SYST), not real MINERvA errors, so no data values are loaded -- only the model.

    srun ... $ADONIS_PY -u -m analysis.paper.minerva_fit
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
# align the 28-knob basis EXACTLY with physfit_gate1 (same order, priors, SYST, N_BINS, binning)
from analysis.paper.physical_fit import (
    knobs_of, theta_nominal, PRIOR, PNAMES, NPAR, SYST, N_BINS, design_edges,
)

BANKDIR = os.environ.get("ADONIS_MINERVA_BANK", "output/paper_banks_p4/nu_MINERvA_C/merged")
LABEL = os.environ.get("ADONIS_LABEL", "physfit_minerva")

# ---- MINERvA CC0pi-Np acceptance (Cai et al. 2018 / NUISANCE MINERvA_CC0pinp) --------------------- #
COS20 = float(np.cos(np.deg2rad(20.0)))     # muon forward cut, theta_mu < 20 deg
COS70 = float(np.cos(np.deg2rad(70.0)))     # leading-proton forward cut, theta_p < 70 deg
MU_WIN = (1500.0, 10000.0)                  # p_mu [MeV] -- NUISANCE isCC0piNp_MINERvA_STV caps at 10 GeV
P_WIN = (450.0, 1200.0)                     # leading-proton |p| [MeV]
_MESONS = list(PDG_MESONS)                  # np.isin needs a list -- a frozenset matches NOTHING
# per-nucleon 1e-38 conversion (carbon), identical to info_content.load_cc0pi("dat")'s conv; MeV-binned
# momentum observables display per GeV/c -> x1000 (same convention as physical_fit's pmu).  NB: for a
# carbon bank (offset=0) this scale is common to J, central and MC-err in every bin, so it CANCELS in
# J/sigma and does NOT affect the Fisher/Gate-I result -- it only sets display units of central/sigma.
CONV0 = 1e-33 / 12.0 * 1e38

# bounded observable domains (full physical range, no overflow fold); others use [min, p99] + fold.
DOMAINS = {"mnv_dat": (0.0, 180.0)}


def minerva_signal(B):
    """MINERvA CC0pi-Np selection as a FULL-LENGTH boolean mask + full-length observables.

    Mirrors adonis.workflow.selection.bank_signal's CC0pi branch (proton_lead='global',
    proton_count='ge1', pion_id='none') with the MINERvA windows, but keeps arrays at bank length so
    the Jacobian engine can index per-event weights.  Observables via the validated bank_plot CC1pi
    STV formulas with the pion 4-vector set to zero (they reduce EXACTLY to the CC0pi STV)."""
    mu = B["k_lep"].astype(np.float64)
    pmu = np.linalg.norm(mu[:, 1:], axis=1)
    cmu = mu[:, 3] / np.maximum(pmu, 1e-9)
    n_meson = BP._event_sum(B, np.isin(B["fs_pid"], _MESONS).astype(float))   # zero final-state mesons
    lead, hasp = BP.leading_proton(B)                                          # global leading proton
    pl = np.linalg.norm(lead[:, 1:], axis=1)
    cl = lead[:, 3] / np.maximum(pl, 1e-9)
    mask = (hasp & (n_meson == 0)
            & (pmu >= MU_WIN[0]) & (pmu <= MU_WIN[1]) & (cmu > COS20)
            & (pl > P_WIN[0]) & (pl < P_WIN[1]) & (cl > COS70))
    pip = np.zeros_like(mu)
    dat = np.degrees(np.asarray(BP.dat_1pi(mu, lead, pip)))     # delta_alphaT [deg]
    pn = np.asarray(BP.pN_1pi(mu, lead, pip))                   # p_n^recon [MeV]
    dpt = np.asarray(BP.dpt_1pi(mu, lead, pip))                 # delta_pT [MeV]
    return mask, {"mnv_dat": dat, "mnv_pn": pn, "mnv_dpt": dpt}


def build_minerva_datasets(B, w0, log):
    """3 observables (dat/pn/dpt), uniform bins (design_edges; dat bounded 0-180, pn/dpt [min,p99]+fold),
    diagonal syst+MC error model, Asimov centrals.  Carbon -> no free-H offset."""
    mask, obs = minerva_signal(B)
    nsel = int(mask.sum())
    log(f"  MINERvA CC0pi-Np selection: {nsel} / {len(w0)} events pass ({nsel/len(w0):.2%})")
    obs_defs = [
        ("MINERvA dat", "mnv_dat", obs["mnv_dat"], CONV0),           # [deg]
        ("MINERvA pn",  "mnv_pn",  obs["mnv_pn"],  CONV0 * 1000.0),  # [MeV] momentum -> per-GeV display
        ("MINERvA dpt", "mnv_dpt", obs["mnv_dpt"], CONV0 * 1000.0),  # [MeV] momentum -> per-GeV display
    ]
    ds = []
    for name, dkey, vals, base_conv in obs_defs:
        v = vals[mask]
        dom = DOMAINS.get(dkey)
        edges = design_edges(v, N_BINS, domain=dom)               # NUISANCE edges unavailable -> design
        nb = len(edges) - 1; bw = np.diff(edges)
        eps = (edges[-1] - edges[0]) * 1e-12
        vc = np.clip(vals, edges[0] + eps, edges[-1] - eps)       # overflow fold into first/last bin
        sel, bidx, nbA = IC._bin(mask, vc, edges); assert nbA == nb
        d = dict(name=name, key=dkey, sel_idx=sel, binidx=bidx, nbin=nb,
                 scale_bin=base_conv / bw, offset=np.zeros(nb), edges=edges)
        central = IC.bin_w(d, w0)                                  # Asimov central
        sw2 = np.bincount(bidx, weights=np.asarray(w0)[sel]**2, minlength=nb)
        mcerr = d["scale_bin"] * np.sqrt(sw2)
        var = (SYST * central)**2 + mcerr**2
        d.update(data=central, sigma=np.sqrt(var), mcerr=mcerr)
        ds.append(d)
        med_mc = np.median(mcerr / np.maximum(central, 1e-30))
        med_tot = np.median(np.sqrt(var) / np.maximum(central, 1e-30))
        log(f"  [{name}] {nb} bins over [{edges[0]:.3g},{edges[-1]:.3g}] {'(0-180 full)' if dom else '(min..p99+fold)'}"
            f" | MC err med {med_mc:.2%} | total err med {med_tot:.2%}")
    return ds


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    B = BP.load_bank(BANKDIR); JB = BR.to_jax(B); grids = BR.default_grids(); nom = nominal_knobs()
    w0 = np.asarray(BR.weight_jit(JB, nom, grids))
    log(f"bank {BANKDIR}: {len(w0)} events | {NPAR} knobs | {N_BINS} bins/obs | syst {SYST:.0%}")
    log(f"CUTS  p_mu[{MU_WIN[0]:.0f},{MU_WIN[1]:.0f}]MeV cos_mu>{COS20:.4f}(20deg) | "
        f"p_lead({P_WIN[0]:.0f},{P_WIN[1]:.0f})MeV cos_p>{COS70:.4f}(70deg) | 0 mesons, >=1 proton (global,ge1)")
    ds = build_minerva_datasets(B, w0, log)
    nbins = sum(d["nbin"] for d in ds)
    log(f"{len(ds)} datasets, {nbins} bins  |  bin edges: design_edges (NO NUISANCE STV file exists)")

    # ---- Jacobian at nominal: one jvp per knob (same engine as physical_fit) ----------------------- #
    th0 = theta_nominal(nom)
    def wf(theta, JB):
        return BR.bank_weight(JB, knobs_of(theta, nom), grids)
    wf_jit = jax.jit(wf)
    jvp_wf = jax.jit(lambda th, tang, JB: jax.jvp(lambda t: wf(t, JB), (th,), (tang,))[1])

    # verification (1): nominal identity -- the jvp's linearization point wf(th0)=bank_weight(nom) must
    # reproduce weight_jit(nom), so the binned central IC.bin_w(d,w0) is the Jacobian's central.
    w0b = np.asarray(wf_jit(jnp.asarray(th0), JB))
    id_err = float(np.max(np.abs(w0b - w0)))
    cen_err = max(float(np.max(np.abs(IC.bin_w(d, w0b) - IC.bin_w(d, w0)))) for d in ds)
    log(f"(V1) nominal identity |w(th0)-w_nominal|_max = {id_err:.2e} ; max |central diff| = {cen_err:.2e}")

    J = np.zeros((nbins, NPAR))
    row0 = np.cumsum([0] + [d["nbin"] for d in ds])
    for k in range(NPAR):
        g = np.asarray(jvp_wf(jnp.asarray(th0), jnp.zeros(NPAR).at[k].set(1.0), JB))
        for j, d in enumerate(ds):
            J[row0[j]:row0[j+1], k] = IC.bin_w0(d, g)
        log(f"  jvp {k+1:2d}/{NPAR} {PNAMES[k]}")
    sigma = np.concatenate([d["sigma"] for d in ds])

    # ---- Gate I: Asimov Fisher + priors (recomputed on THIS sample) -------------------------------- #
    Jw = J / sigma[:, None]
    F = Jw.T @ Jw
    V = np.linalg.inv(F + np.diag(1.0 / PRIOR**2))
    sig_post = np.sqrt(np.diag(V))
    shrink = sig_post / PRIOR
    verdict = np.where(shrink < 0.5, "FIT", "freeze")
    reach = np.sqrt(np.maximum(np.diag(F), 0.0))                   # per-knob Fisher reach sqrt(F_kk)
    order = np.argsort(shrink)
    print(f"\n==== MINERvA CC0pi-Np GATE I (Asimov Fisher, {nbins} bins, syst {SYST:.0%}) ====")
    print(f"{'knob':>18} {'prior':>7} {'sqrtFkk':>10} {'sig_post':>9} {'shrink':>7}  verdict")
    for k in order:
        print(f"{PNAMES[k]:>18} {PRIOR[k]:7.2f} {reach[k]:10.3g} {sig_post[k]:9.3f} {shrink[k]:7.2f}  {verdict[k]}")
    nfit = int((shrink < 0.5).sum())
    print(f"\n{nfit}/{NPAR} knobs pass Gate I (shrinkage < 0.5) on MINERvA alone")

    # verification (3): the physically-active knobs must be NONZERO (weak probe informs hard vertex + FSI)
    print("\n---- verification (3): reach sqrt(F_kk) of physically-active knobs (must be > 0) ----")
    for name in ["M_A_qe", "vector_strength", "kF_sf", "sabs", "s_NN_elastic[0]",
                 "s_NN_inelastic[1]", "f_NN_cex", "qe_norm", "res_norm"]:
        k = PNAMES.index(name)
        print(f"    {name:>18}  sqrtF={reach[k]:.4g}   {'OK' if reach[k] > 0 else 'ZERO!!'}")

    os.makedirs("output/altgen", exist_ok=True)
    outpath = f"output/altgen/{LABEL}.npz"
    np.savez(outpath, J=J, sigma=sigma, F=F, V=V, prior=PRIOR,
             sig_post=sig_post, shrink=shrink, pnames=PNAMES, nbins=nbins,
             row0=row0, dsnames=[d["name"] for d in ds], dskeys=[d["key"] for d in ds],
             **{f"{d['key']}_edges": d["edges"] for d in ds},
             **{f"{d['key']}_central": d["data"] for d in ds},
             **{f"{d['key']}_sigma": d["sigma"] for d in ds})
    log(f"[out] {outpath}")
    log("done")


if __name__ == "__main__":
    main()
