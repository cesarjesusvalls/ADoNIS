"""PHYSICAL FIT (logbook 17) — step 1: Gate I (Asimov Fisher informativeness).

Program: a fit whose parameter motion is accepted ONLY when physically informed.
  Gate I  (this run): with uncorrelated priors on all knobs, which does the sample actually inform?
          Asimov at nominal: posterior V = (J^T C^-1 J + Pi^-1)^-1. FIT if sigma_post < 0.5*prior.
  Gate II (next): coherence — per-knob Cochran's Q on per-bin demands + vector split-fit Q_split.
  Flag    (next): post-fit residual regions no coherent knob release fixes.

Config (methodology primary, logbook 17): 5 observables — CC0pi dpt/dat (1e-38/nucleon) +
CC1pi pN/dpTT/daT (nb/CH incl. frozen free-H) — each with N_BINS uniform bins over [min, p99]
(overflow folded), DIAGONAL error model sigma^2 = (SYST*d)^2 + ADoNIS-MC^2 (stat of a 3M-scale
sample is 0.5-0.9%, negligible vs SYST=5%; ADoNIS-MC folded in per 'use the ADoNIS we have').

Knob set: full17 (info_content PSETS) + norm/strength/FF knobs = 27. Priors: 20% multiplicative
(nominal-1 knobs); natural units for non-multiplicative: Eb_shift +/-4 MeV, f_NN_cex +/-0.1.

Outputs: gate1 table (knob, prior, sigma_post, shrinkage, verdict) + npz (J, sigma, F, V, specs).
Live progress: python -u -> logfile.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from analysis.t2k.differentiability import bank_plot as BP, bank_reweight as BR
from analysis.t2k.differentiability.full_knobs import nominal_knobs
from analysis.t2k.differentiability import info_content as IC

BANKDIR = os.environ.get("ADONIS_EVENT_BANK", "output/event_bank")
N_BINS = int(os.environ.get("ADONIS_NBINS", "20"))
SYST = float(os.environ.get("ADONIS_SYST", "0.05"))
LABEL = os.environ.get("ADONIS_LABEL", "physfit_gate1")

# ---- knob spec: enumerated through the SINGLE SOURCE OF TRUTH (full_knobs.knob_specs), which drops
# sscat (dead) and pw_norm (dormant DCC infra) and expands the tuple knobs. We attach only the fit-local
# PRIOR policy here: 20% multiplicative default; natural units for the non-multiplicative knobs. -------- #
from analysis.t2k.differentiability.full_knobs import knob_specs as _knob_specs
_PRIOR_POLICY = {"Eb_shift": 4.0, "f_NN_cex": 0.10}     # everything else: 20% multiplicative
SPEC = [(name, idx, lab, _PRIOR_POLICY.get(name, 0.20))
        for (name, idx, lab, _nom) in _knob_specs(nominal_knobs())]
NPAR = len(SPEC)
PNAMES = [f"{k}[{i}]" if i is not None else k for k, i, *_ in SPEC]
PRIOR = np.array([s[3] for s in SPEC])


def theta_nominal(nom):
    th = np.zeros(NPAR)
    for j, (key, idx, *_ ) in enumerate(SPEC):
        v = nom[key]
        th[j] = float(v[idx]) if idx is not None else float(v)
    return th


def knobs_of(theta, nom):
    """Full knob dict from the SPEC theta vector (tuple components rebuilt)."""
    k = dict(nom)
    tup = {}
    for j, (key, idx, *_ ) in enumerate(SPEC):
        if idx is None:
            k[key] = theta[j]
        else:
            tup.setdefault(key, list(np.asarray(nom[key])))[idx] = theta[j]
    for key, lst in tup.items():
        k[key] = jnp.asarray(lst)
    return k


def design_edges(values, n_bins, p_hi=99.0, domain=None):
    """Uniform edges. Bounded observables (domain=(lo,hi), e.g. dat on [0,pi]) are binned over the
    FULL physical range with no overflow fold — folding a percentile tail into the last bin inflates
    its dsigma/dx by the width mismatch (visible last-bin jump). Unbounded ones use [min, p99] with
    the tail folded into the last bin (a genuine overflow-bin convention)."""
    v = np.asarray(values)
    if domain is not None:
        edges = np.linspace(domain[0], domain[1], n_bins + 1)
    else:
        edges = np.unique(np.linspace(float(v.min()), float(np.percentile(v, p_hi)), n_bins + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    return edges


# bounded observables: full physical range, no overflow fold
DOMAINS = {"dat": (0.0, float(np.pi)), "daT": (0.0, 180.0)}


def build_physfit_datasets(B, w0, log):
    """9 observables, uniform bins (p99+overflow fold, or full range for bounded), diagonal syst+MC
    errors, Asimov centrals. Muon/pion kinematics carry the Q^2 information the STV variables
    integrate out. NB: the new CC1pi kinematics (ppi/cospi) get offset=0 (no free-H term) — valid
    for closure-type modes where data and model share the bank (any offset cancels identically);
    NOT valid for GENIE mode without extending freeH_offsets."""
    ds = []
    lead0, _ = BP.leading_proton(B); sig0 = BP.signal_cc0pi(B)[0]
    kmu = B["k_mu"].astype(np.float64)
    mask1, lead1, pip1 = BP.signal_cc1pi_stv(B)
    # muon kinematics (junk zero-weight padding rows -> harmless clipped values)
    kmu_ok = np.where(np.isfinite(kmu) & (np.abs(kmu) < 1e6), kmu, 0.0)
    pmu = np.sqrt(np.sum(kmu_ok[:, 1:]**2, axis=1))
    cmu = kmu_ok[:, 3] / np.maximum(pmu, 1e-9)
    # pion kinematics (CC1pi single pi+)
    pip_ok = np.where(np.isfinite(pip1) & (np.abs(pip1) < 1e6), pip1, 0.0)
    ppi = np.sqrt(np.sum(np.asarray(pip_ok)[:, 1:]**2, axis=1))
    cpi = np.asarray(pip_ok)[:, 3] / np.maximum(ppi, 1e-9)
    obs_defs = [
        ("CC0pi dpt",   sig0,  np.asarray(BP.dpt(B, lead0)),  "dpt",   "cc0pi"),
        ("CC0pi dat",   sig0,  np.asarray(BP.dat(B, lead0)),  "dat",   "cc0pi"),
        ("CC0pi pmu",   sig0,  pmu,                           "pmu",   "cc0pi"),
        ("CC0pi cosmu", sig0,  cmu,                           "cosmu", "cc0pi"),
        ("CC1pi pN",    mask1, np.asarray(BP.pN_1pi(kmu, lead1, pip1)),               "pn",    "cc1pi"),
        ("CC1pi dpTT",  mask1, np.asarray(BP.dptt_1pi(kmu, lead1, pip1)),             "dptt",  "cc1pi"),
        ("CC1pi daT",   mask1, np.degrees(np.asarray(BP.dat_1pi(kmu, lead1, pip1))),  "daT",   "cc1pi"),
        ("CC1pi ppi",   mask1, ppi,                           "ppi",   "cc1pi"),
        ("CC1pi cospi", mask1, cpi,                           "cospi", "cc1pi"),
    ]
    # free-H offsets on OUR edges (theta-independent; enters centrals -> syst sigma, cancels in J)
    edges_by = {}
    for name, mask, vals, dkey, chan in obs_defs:
        v = vals[np.asarray(mask, bool)]
        dom = DOMAINS.get(dkey)
        if dkey in ("cosmu", "cospi"):                  # bounded above at 1, cut below by acceptance
            dom = (float(v.min()), 1.0)
        edges_by[dkey] = design_edges(v, N_BINS, domain=dom)
    fH = IC.freeH_offsets({k: v for k, v in edges_by.items() if k in ("pn", "dptt", "daT")})
    for name, mask, vals, dkey, chan in obs_defs:
        edges = edges_by[dkey]; nb = len(edges) - 1; bw = np.diff(edges)
        eps = (edges[-1] - edges[0]) * 1e-12
        vc = np.clip(vals, edges[0] + eps, edges[-1] - eps)      # overflow fold
        sel, bidx, nbA = IC._bin(mask, vc, edges); assert nbA == nb
        if chan == "cc0pi":
            if dkey in ("dpt", "dat"):
                _, conv, _, _ = IC.load_cc0pi(dkey)
            else:
                # base 1e-38/nucleon per NATIVE unit (dat conv is per rad = per native);
                # pmu binned in MeV but displayed per GeV/c -> x1000 (same convention as dpt)
                _, conv0, _, _ = IC.load_cc0pi("dat")
                conv = conv0 * (1000.0 if dkey == "pmu" else 1.0)
            scale_bin = conv / bw
            offset = np.zeros(nb)
        else:
            scale_bin = 1.0 / bw
            offset = fH[dkey] if dkey in fH else np.zeros(nb)   # new CC1pi kin: no free-H (closure-safe)
        d = dict(name=name, key=dkey, sel_idx=sel, binidx=bidx, nbin=nb, scale_bin=scale_bin,
                 offset=offset, edges=edges)
        central = IC.bin_w(d, w0)                                 # Asimov central (incl. free-H)
        sw2 = np.bincount(bidx, weights=np.asarray(w0)[sel]**2, minlength=nb)
        mcerr = scale_bin * np.sqrt(sw2)
        var = (SYST * central)**2 + mcerr**2
        d.update(data=central, sigma=np.sqrt(var), Cinv=np.diag(1.0 / np.maximum(var, 1e-300)),
                 mcerr=mcerr)
        ds.append(d)
        log(f"  [{name}] {nb} bins | MC err med {np.median(mcerr/np.maximum(central,1e-30)):.1%} "
            f"| total err med {np.median(np.sqrt(var)/np.maximum(central,1e-30)):.1%}")
    return ds


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    B = BP.load_bank(BANKDIR); JB = BR.to_jax(B); grids = BR.default_grids(); nom = nominal_knobs()
    w0 = np.asarray(BR.weight_jit(JB, nom, grids))
    log(f"bank {len(w0)} events | {NPAR} knobs | {N_BINS} bins/obs | syst {SYST:.0%}")
    ds = build_physfit_datasets(B, w0, log)
    nbins = sum(d["nbin"] for d in ds)
    log(f"{len(ds)} datasets, {nbins} bins")

    # ---- Jacobian at nominal: one jvp per knob (info_content pattern; JB as argument) ------------- #
    th0 = theta_nominal(nom)
    def wf(theta, JB):
        return BR.bank_weight(JB, knobs_of(theta, nom), grids)
    jvp_wf = jax.jit(lambda th, tang, JB: jax.jvp(lambda t: wf(t, JB), (th,), (tang,))[1])
    J = np.zeros((nbins, NPAR))
    row0 = np.cumsum([0] + [d["nbin"] for d in ds])
    for k in range(NPAR):
        g = np.asarray(jvp_wf(jnp.asarray(th0), jnp.zeros(NPAR).at[k].set(1.0), JB))
        for j, d in enumerate(ds):
            J[row0[j]:row0[j+1], k] = IC.bin_w0(d, g)
        log(f"  jvp {k+1:2d}/{NPAR} {PNAMES[k]}")
    sigma = np.concatenate([d["sigma"] for d in ds])

    # ---- Gate I: Asimov Fisher + priors ----------------------------------------------------------- #
    Jw = J / sigma[:, None]
    F = Jw.T @ Jw                                              # data Fisher
    V = np.linalg.inv(F + np.diag(1.0 / PRIOR**2))             # posterior (marginalized)
    sig_post = np.sqrt(np.diag(V))
    shrink = sig_post / PRIOR
    verdict = np.where(shrink < 0.5, "FIT", "freeze")
    order = np.argsort(shrink)
    print(f"\n==== GATE I (Asimov Fisher, {nbins} bins, syst {SYST:.0%}, priors: 20% mult / natural) ====")
    print(f"{'knob':>16} {'prior':>7} {'sig_post':>9} {'shrink':>7}  verdict")
    for k in order:
        print(f"{PNAMES[k]:>16} {PRIOR[k]:7.2f} {sig_post[k]:9.3f} {shrink[k]:7.2f}  {verdict[k]}")
    nfit = int((shrink < 0.5).sum())
    print(f"\n{nfit}/{NPAR} knobs pass Gate I (shrinkage < 0.5)")

    os.makedirs("output/altgen", exist_ok=True)
    np.savez(f"output/altgen/{LABEL}.npz", J=J, sigma=sigma, F=F, V=V, prior=PRIOR,
             sig_post=sig_post, shrink=shrink, pnames=PNAMES, nbins=nbins,
             row0=row0, dsnames=[d["name"] for d in ds],
             **{f"{d['key']}_edges": d["edges"] for d in ds},
             **{f"{d['key']}_central": d["data"] for d in ds},
             **{f"{d['key']}_sigma": d["sigma"] for d in ds})
    log(f"[out] output/altgen/{LABEL}.npz")
    log("done")


if __name__ == "__main__":
    main()
