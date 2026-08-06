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

Knob set: full17 (info_content PSETS) + norm/strength/FF knobs = 28 (= NPAR). Priors: 20% multiplicative
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

from adonis.reweight import bank_plot as BP, bank_reweight as BR
from adonis.reweight.reweight_model import nominal_knobs
from analysis.paper import info_content as IC
from analysis.paper import fisher_engine as FE

BANKDIR = os.environ.get("ADONIS_EVENT_BANK", "output/event_bank")
N_BINS = int(os.environ.get("ADONIS_NBINS", "20"))
SYST = float(os.environ.get("ADONIS_SYST", "0.05"))
OBS = os.environ.get("PHYSFIT_OBS", "all")               # observable subset (see OBS_SUBSETS)
LABEL = os.environ.get("ADONIS_LABEL", "physfit_gate1" if OBS == "all" else f"physfit_gate1_{OBS}")
# CH target for T2K CC1pi+: add the reweightable free-H (nu_mu p -> mu- p pi+) GRADIENT to pN/dpTT/daT.
# The central already carries free-H via the frozen freeH_offsets; only the gradient was carbon-only.
T2K_CH = os.environ.get("ADONIS_T2K_CH", "") == "1"
T2K_H_BANK = os.environ.get("ADONIS_T2K_H_BANK", "output/paper_banks_p4/nu_T2K_H/merged")
# Stream the jvp over chunk files (peak memory = one chunk) instead of loading the whole bank onto the
# device -> lets the 20M-event Jacobian run on the GPU.  Bit-identical to the whole-bank path.
CHUNK_JVP = os.environ.get("ADONIS_CHUNK_JVP", "") == "1"

# ---- knob spec: enumerated through the SINGLE SOURCE OF TRUTH (full_knobs.knob_specs), which drops
# sscat (dead) and pw_norm (dormant DCC infra) and expands the tuple knobs. We attach only the fit-local
# PRIOR policy here: 20% multiplicative default; natural units for the non-multiplicative knobs. -------- #
from adonis.reweight.reweight_model import knob_specs as _knob_specs
_PRIOR_POLICY = {"Eb_shift": 4.0, "f_NN_cex": 0.10}     # everything else: 20% multiplicative
SPEC = [(name, idx, lab, _PRIOR_POLICY.get(name, 0.20))
        for (name, idx, lab, _nom) in _knob_specs(nominal_knobs())]
NPAR = len(SPEC)
PNAMES = [f"{k}[{i}]" if i is not None else k for k, i, *_ in SPEC]
PRIOR = np.array([s[3] for s in SPEC])


def theta_nominal(nom):
    th = np.zeros(NPAR)
    for j, (key, idx, *_ ) in enumerate(SPEC):
        v = getattr(nom, key)
        th[j] = float(v[idx]) if idx is not None else float(v)
    return th


def knobs_of(theta, nom):
    """Full PhysicsParams from the SPEC theta vector (tuple components rebuilt)."""
    upd = {}
    tup = {}
    for j, (key, idx, *_ ) in enumerate(SPEC):
        if idx is None:
            upd[key] = theta[j]
        else:
            tup.setdefault(key, list(np.asarray(getattr(nom, key))))[idx] = theta[j]
    for key, lst in tup.items():
        upd[key] = jnp.asarray(lst)
    return nom._replace(**upd)


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

# ---- observed multiplicities (CC-inclusive) ------------------------------------------------------- #
# Sample: CC inclusive under the T2K MUON acceptance only (p_mu > MU_LO, cos_mu > COSMU) -- NOT
# bank_plot.acceptance(), which is the CC0pi signal cut and carries a leading-proton window that would
# make N_p = 0 unreachable.  The exclusive selections cannot host these: N_pi+- is 0 by construction in
# CC0pi and 1 in CC1pi, so the multiplicities only carry information in the inclusive sample.
#   N_p     : protons above P_THR_MULT (detection threshold), no upper bound
#   N_pi+-  : pi+ and pi- counted together in one sample, no threshold
# Integer edges; the top bin is an overflow (the generic clip in build_physfit_datasets folds it).
# Occupancies at nominal (1.87M bank): N_p 22/63/11/3/1%, N_pi+- 85/15/0.1%.
P_THR_MULT = float(os.environ.get("ADONIS_PMULT_THR", "300.0"))     # MeV/c, proton detection threshold
MULT_EDGES = {
    "n_p":     np.arange(-0.5, 5.0, 1.0),      # 0,1,2,3,>=4
    "n_chpi":  np.arange(-0.5, 3.0, 1.0),      # 0,1,>=2
}

# ---- observable subsets (paper section 3: "what is worth fitting, per observable class") ----------
# Single source of truth for BOTH the per-subset Gate-I Fisher and any subset-restricted fit: values
# are dkeys of build_physfit_datasets' obs_defs.  "kin9" is the frozen 9-observable suite whose J is
# persisted (physfit_gate1.npz); every kinematic subset is a ROW SLICE of that J -- no bank pass.
# "mult" adds rows, so "full" requires its own Gate-I run.  ("all" = kin9, kept for back-compat with
# the persisted p9_*/physfit_* labels -- do NOT redefine it to include the multiplicities.)
OBS_SUBSETS = {
    "lepton":    ["pmu", "cosmu"],                                  # what a lepton-only analysis sees
    "leptonhad": ["pmu", "cosmu", "ppi", "cospi"],                  # + the hadron (pion) kinematics
    "tki":       ["dpt", "dat", "pn", "dptt", "daT"],               # transverse-kinematic imbalance
    "mult":      ["n_p", "n_chpi"],                                 # observed multiplicities (inclusive)
    "kin9":      ["dpt", "dat", "pmu", "cosmu", "pn", "dptt", "daT", "ppi", "cospi"],
}
OBS_SUBSETS["all"] = list(OBS_SUBSETS["kin9"])                      # back-compat alias
OBS_SUBSETS["full"] = OBS_SUBSETS["kin9"] + OBS_SUBSETS["mult"]     # everything (needs its own run)


def _physfit_obs_defs(B):
    """(name, mask, vals, dkey, chan) for every physfit observable on bank B.  Depends ONLY on B, so the
    chunked Jacobian can re-derive it per chunk (whole bank OR one chunk file)."""
    lead0, _ = BP.leading_proton(B); sig0 = BP.signal_cc0pi(B)[0]
    kmu = B["k_lep"].astype(np.float64)
    mask1, lead1, pip1 = BP.signal_cc1pi_stv(B)
    kmu_ok = np.where(np.isfinite(kmu) & (np.abs(kmu) < 1e6), kmu, 0.0)      # junk padding -> clipped
    pmu = np.sqrt(np.sum(kmu_ok[:, 1:]**2, axis=1)); cmu = kmu_ok[:, 3] / np.maximum(pmu, 1e-9)
    pip_ok = np.where(np.isfinite(pip1) & (np.abs(pip1) < 1e6), pip1, 0.0)   # CC1pi single pi+
    ppi = np.sqrt(np.sum(np.asarray(pip_ok)[:, 1:]**2, axis=1)); cpi = np.asarray(pip_ok)[:, 3] / np.maximum(ppi, 1e-9)
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
    # observed multiplicities, CC-inclusive under the MUON acceptance only (see MULT_EDGES above).
    _T = BP._tune()
    inc = (pmu > _T.MU_LO) & (cmu > _T.COSMU)
    npip, _npi0, npim = BP.pion_counts(B)
    obs_defs += [
        ("Incl N_p",     inc, BP.n_protons(B, pmin=P_THR_MULT).astype(float), "n_p",    "incl"),
        ("Incl N_pi+-",  inc, (npip + npim).astype(float),                    "n_chpi", "incl"),
    ]
    return obs_defs


def _physfit_scale_bin(dkey, chan, bw):
    """Per-bin conversion (theta- and bank-INDEPENDENT).  CC0pi/incl -> NUISANCE 1e-38/nucleon conv
    (pmu displayed per GeV/c -> x1000); CC1pi -> nb/unit (1/bw)."""
    if chan == "cc0pi":
        if dkey in ("dpt", "dat"):
            _, conv, _, _ = IC.load_cc0pi(dkey)
        else:
            _, conv0, _, _ = IC.load_cc0pi("dat"); conv = conv0 * (1000.0 if dkey == "pmu" else 1.0)
        return conv / bw
    if chan == "incl":
        _, conv0, _, _ = IC.load_cc0pi("dat"); return conv0 / bw
    return 1.0 / bw


def _physfit_bin_ds(B, edges_by, dkeys):
    """Per-obs dataset dicts (sel_idx/binidx/nbin/scale_bin) for bank B on the FIXED edges_by -- exactly
    the fields IC.bin_w0 needs, for the (chunked) Jacobian.  NO central/sigma/offset.  The chunked jvp
    calls this per chunk; build_physfit_datasets calls it once on the whole bank and adds central/sigma."""
    defs = {d[3]: d for d in _physfit_obs_defs(B)}
    ds = []
    for dkey in dkeys:
        name, mask, vals, _, chan = defs[dkey]
        edges = edges_by[dkey]; nb = len(edges) - 1; bw = np.diff(edges)
        eps = (edges[-1] - edges[0]) * 1e-12
        vc = np.clip(vals, edges[0] + eps, edges[-1] - eps)      # overflow fold
        sel, bidx, nbA = IC._bin(mask, vc, edges); assert nbA == nb
        ds.append(dict(name=name, key=dkey, sel_idx=sel, binidx=bidx, nbin=nb,
                       scale_bin=_physfit_scale_bin(dkey, chan, bw)))
    return ds


def physfit_edges(B, obs=None):
    """design_edges per observable over the WHOLE bank (fixes the binning every chunk shares) + the
    free-H offsets.  Returns (edges_by, fH, dkeys)."""
    obs_defs = _physfit_obs_defs(B)
    if obs is not None:
        keep = set(obs); unknown = keep - {d[3] for d in obs_defs}
        if unknown:
            raise KeyError(f"unknown observable key(s): {sorted(unknown)}")
        obs_defs = [d for d in obs_defs if d[3] in keep]
    edges_by = {}
    for name, mask, vals, dkey, chan in obs_defs:
        if dkey in MULT_EDGES:                          # integer multiplicity bins (top bin = overflow)
            edges_by[dkey] = MULT_EDGES[dkey]; continue
        v = vals[np.asarray(mask, bool)]; dom = DOMAINS.get(dkey)
        if dkey in ("cosmu", "cospi"):                  # bounded above at 1, cut below by acceptance
            dom = (float(v.min()), 1.0)
        edges_by[dkey] = design_edges(v, N_BINS, domain=dom)
    fH_edges = {k: v for k, v in edges_by.items() if k in ("pn", "dptt", "daT")}
    fH = IC.freeH_offsets(fH_edges) if fH_edges else {}   # skip the free-H generation when unused
    return edges_by, fH, [d[3] for d in obs_defs]


def build_physfit_datasets(B, w0, log, obs=None):
    """Full datasets (sel/binidx/scale_bin + Asimov central, sigma, Cinv, offset) over the WHOLE bank.
    The binning (sel/binidx/scale_bin) comes from the SAME _physfit_bin_ds the chunked Jacobian uses; this
    adds the whole-bank central/sigma + the frozen free-H offset (theta-independent -> cancels in J).
    obs: dkeys to keep (None = all)."""
    edges_by, fH, dkeys = physfit_edges(B, obs)
    binds = _physfit_bin_ds(B, edges_by, dkeys)
    ds = []
    for bd in binds:
        offset = fH.get(bd["key"], np.zeros(bd["nbin"]))         # free-H only on pn/dptt/daT; else zeros
        d = dict(bd, offset=offset, edges=edges_by[bd["key"]])
        central = IC.bin_w(d, w0)                                 # Asimov central (incl. frozen free-H)
        sw2 = np.bincount(d["binidx"], weights=np.asarray(w0)[d["sel_idx"]]**2, minlength=d["nbin"])
        mcerr = d["scale_bin"] * np.sqrt(sw2)
        var = (SYST * central)**2 + mcerr**2                      # kept for Cinv + the total-error log
        d.update(data=central, sigma=FE.bin_sigma(central, mcerr, SYST),  # empty-bin guarded
                 Cinv=np.diag(1.0 / np.maximum(var, 1e-300)), mcerr=mcerr)
        ds.append(d)
        log(f"  [{d['name']}] {d['nbin']} bins | MC err med {np.median(mcerr/np.maximum(central,1e-30)):.1%} "
            f"| total err med {np.median(np.sqrt(var)/np.maximum(central,1e-30)):.1%}")
    return ds


def _stream_selected(bank_dir, obs_fn, log=None):
    """Stream a bank ONE CHUNK AT A TIME; accumulate only the SELECTED (signal) events' (value, nominal
    weight) per observable.  Peak memory = one chunk (~76MB) of PURE NUMPY: no jax array is ever created
    here, so Pass 1 cannot accumulate device buffers.  The nominal per-event weight is the bank's stored w0
    (the same weight bank_plot's forward histogram uses), NOT weight_jit(nom) -- the two agree to the
    reduced-quadratic nominal-identity roundoff (~1e-7), so central/sigma match the whole-bank path
    negligibly, and the Jacobian (built separately by the pass-2 jvp) is unaffected.  The accumulator is the
    signal sample (~0.1-1% of the bank), never the bank.  obs_fn(B_chunk) -> iterable of
    (name, mask, vals, dkey, chan).

    Order-independent: mask/vals/w are per-event, and everything downstream (design_edges = min/percentile;
    central = bincount) is order-independent, so chunk order is irrelevant."""
    import glob
    nch = BP.bank_nchunks(bank_dir); files = sorted(glob.glob(f"{bank_dir}/chunk_*.npz"))
    acc = {}
    for ci, f in enumerate(files):
        Bc = BP.load_bank_chunk(f, nch)
        w0c = np.asarray(Bc["w0"])                              # stored nominal per-event weight (numpy)
        for name, mask, vals, dkey, chan in obs_fn(Bc):
            m = np.asarray(mask, bool)
            a = acc.setdefault(dkey, dict(name=name, chan=chan, vals=[], w=[]))
            a["vals"].append(np.asarray(vals)[m]); a["w"].append(w0c[m])
        del Bc, w0c
        if log is not None and ((ci + 1) % 20 == 0 or ci + 1 == len(files)):
            log(f"  scan {ci + 1}/{len(files)} chunks")
    return {k: dict(name=v["name"], chan=v["chan"],
                    vals=np.concatenate(v["vals"]) if v["vals"] else np.zeros(0),
                    w=np.concatenate(v["w"]) if v["w"] else np.zeros(0)) for k, v in acc.items()}


def _central_from_selected(vals, w, edges, scale_bin, offset):
    """Per-bin dsigma/dx + MC error from an accumulated (selected value, nominal weight) sample -- the
    streamed twin of IC.bin_w: clip (overflow fold) -> IC._bin -> scale*bincount + offset."""
    nb = len(edges) - 1; eps = (edges[-1] - edges[0]) * 1e-12
    vc = np.clip(vals, edges[0] + eps, edges[-1] - eps)
    sel, bidx, nbA = IC._bin(np.ones(len(vc), bool), vc, edges); assert nbA == nb
    central = scale_bin * np.bincount(bidx, weights=w[sel], minlength=nb) + offset
    mcerr = scale_bin * np.sqrt(np.bincount(bidx, weights=w[sel] ** 2, minlength=nb))
    return central, mcerr


def physfit_stream_datasets(bank_dir, grids, nom, log, obs=None):
    """Full physfit datasets built by STREAMING the bank one chunk at a time (peak = one chunk) -- the 20M
    bank is never concatenated in memory.  Returns the SAME datasets as build_physfit_datasets over the whole
    bank (edges/central/sigma/Cinv/offset), minus the global sel/binidx (the chunked Jacobian recomputes
    those per chunk).  Matches the whole-bank path to the nominal-identity roundoff (~1e-7: the streamed
    nominal weight is the stored w0, see _stream_selected); the Jacobian is built separately and is exact."""
    keep = None if obs is None else set(obs)

    def obs_fn(Bc):
        for t in _physfit_obs_defs(Bc):
            if keep is None or t[3] in keep:
                yield t
    acc = _stream_selected(bank_dir, obs_fn, log=log)
    if keep is not None:
        unknown = keep - set(acc)
        if unknown:
            raise KeyError(f"unknown observable key(s): {sorted(unknown)}")
    dkeys = list(acc)                                            # obs_defs order (first-chunk insertion)
    edges_by = {}
    for k in dkeys:
        if k in MULT_EDGES:                                      # integer multiplicity bins
            edges_by[k] = MULT_EDGES[k]; continue
        v = acc[k]["vals"]; dom = DOMAINS.get(k)
        if k in ("cosmu", "cospi"):
            dom = (float(v.min()), 1.0)
        edges_by[k] = design_edges(v, N_BINS, domain=dom)
    fH_edges = {k: edges_by[k] for k in dkeys if k in ("pn", "dptt", "daT")}
    fH = IC.freeH_offsets(fH_edges) if fH_edges else {}
    ds = []
    for k in dkeys:
        edges = edges_by[k]; nb = len(edges) - 1; bw = np.diff(edges)
        scale_bin = _physfit_scale_bin(k, acc[k]["chan"], bw)
        offset = fH.get(k, np.zeros(nb))
        central, mcerr = _central_from_selected(acc[k]["vals"], acc[k]["w"], edges, scale_bin, offset)
        var = (SYST * central) ** 2 + mcerr ** 2
        ds.append(dict(name=acc[k]["name"], key=k, nbin=nb, scale_bin=scale_bin, offset=offset, edges=edges,
                       data=central, sigma=FE.bin_sigma(central, mcerr, SYST),
                       Cinv=np.diag(1.0 / np.maximum(var, 1e-300)), mcerr=mcerr))
        log(f"  [{acc[k]['name']}] {nb} bins | {len(acc[k]['vals']):,} sig ev | "
            f"MC err med {np.median(mcerr/np.maximum(central,1e-30)):.1%}")
    return ds


def cc1pi_freeH_jacobian(ds, row0, J, grids, nom, log):
    """CH mode: ADD the reweightable free-H (nu_mu p -> mu- p pi+) GRADIENT to the T2K CC1pi+ observables
    (pN, dpTT, daT), in place in J.  The CC1pi+ central already carries free-H via the frozen freeH_offsets
    (theta-independent -> its derivative is zero, so the gradient was carbon-only).  The free-H Jacobian is
    computed on the reweightable nu_T2K_H bank with the SAME per-observable edges, STREAMED per chunk when
    CHUNK_JVP (peak = one H chunk).  daT gets the NUISANCE hydrogen randomization (flat).  Returns per-obs
    free-H central for a consistency check vs freeH_offsets."""
    from adonis.workflow.data_overlay import hydrogen_daT
    cc1 = [(i, d) for i, d in enumerate(ds) if d["key"] in ("pn", "dptt", "daT")]
    if not cc1:
        return {}
    edges_by = {d["key"]: d["edges"] for _, d in cc1}; dkeys = [d["key"] for _, d in cc1]

    def h_obs(BH):                                          # (name, mask, vals, dkey, chan) for H CC1pi+
        kmu = BH["k_lep"].astype(np.float64)
        mask, lead, pip = BP.signal_cc1pi_stv(BH)
        dptt_v = np.asarray(BP.dptt_1pi(kmu, lead, pip))
        daT_v = np.degrees(hydrogen_daT(dptt_v, np.asarray(BP.dat_1pi(kmu, lead, pip)),
                                        np.ones(len(kmu), bool), 0))     # every event free-H -> flat daT
        vmap = {"pn": np.asarray(BP.pN_1pi(kmu, lead, pip)), "dptt": dptt_v, "daT": daT_v}
        return [(f"H {k}", mask, vmap[k], k, "cc1pi") for k in dkeys]

    def h_bin_ds(BH):                                       # per-chunk sel/binidx/scale for the jvp
        defs = {t[3]: t for t in h_obs(BH)}
        out = []
        for k in dkeys:
            _, mask, vals, _, _ = defs[k]
            edges = edges_by[k]; nb = len(edges) - 1; bw = np.diff(edges)
            eps = (edges[-1] - edges[0]) * 1e-12
            vc = np.clip(vals, edges[0] + eps, edges[-1] - eps)
            sel, bidx, nbA = IC._bin(mask, vc, edges); assert nbA == nb
            out.append(dict(key=k, sel_idx=sel, binidx=bidx, nbin=nb, scale_bin=1.0 / bw))
        return out

    # free-H central (for the consistency check) -- streamed selected (value, nominal weight)
    if CHUNK_JVP:
        accH = _stream_selected(T2K_H_BANK, h_obs, log=log)
    else:
        BH = BP.load_bank(T2K_H_BANK); w0H = np.asarray(BH["w0"])        # stored nominal weight (numpy)
        accH = {t[3]: dict(vals=np.asarray(t[2])[np.asarray(t[1], bool)], w=w0H[np.asarray(t[1], bool)])
                for t in h_obs(BH)}
    centralH = {}
    for k in dkeys:
        edges = edges_by[k]; bw = np.diff(edges)
        centralH[k], _ = _central_from_selected(accH[k]["vals"], accH[k]["w"], edges, 1.0 / bw,
                                                np.zeros(len(edges) - 1))

    th0 = theta_nominal(nom)
    def wfH(theta, JB):
        return BR.bank_weight(JB, knobs_of(theta, nom), grids)
    jvp_wfH = jax.jit(lambda th, tang, JB: jax.jvp(lambda t: wfH(t, JB), (th,), (tang,))[1])
    nbins_H = [len(edges_by[k]) - 1 for k in dkeys]
    if CHUNK_JVP:
        JH, row0H = FE.bank_jacobian_chunked(jvp_wfH, th0, T2K_H_BANK, h_bin_ds, nbins_H, NPAR,
                                             log=log, names=PNAMES)
    else:
        JH, row0H = FE.bank_jacobian(jvp_wfH, th0, BR.to_jax(BH), h_bin_ds(BH), NPAR, log=log, names=PNAMES)
    for j, (i, _d) in enumerate(cc1):
        J[row0[i]:row0[i + 1]] += JH[row0H[j]:row0H[j + 1]]             # add free-H gradient to carbon rows
    return centralH


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    obs = OBS_SUBSETS[OBS] if OBS != "all" else None
    grids = BR.default_grids(); nom = nominal_knobs(); th0 = theta_nominal(nom)
    def wf(theta, JB):
        return BR.bank_weight(JB, knobs_of(theta, nom), grids)
    jvp_wf = jax.jit(lambda th, tang, JB: jax.jvp(lambda t: wf(t, JB), (th,), (tang,))[1])
    if CHUNK_JVP:
        # TRUE streaming: neither pass ever holds more than ONE chunk (~76MB).  Pass 1 = datasets
        # (edges/central/sigma) from the selected events only; pass 2 = the per-knob jvp Jacobian.
        # Jacobian is exact (bincount is additive); central/sigma match the whole-bank path to the nominal
        # roundoff (~1e-7: pass 1 uses the stored nominal w0, no jax -> no per-chunk device buffers).
        log(f"streaming {BANKDIR} | {NPAR} knobs | {N_BINS} bins/obs | syst {SYST:.0%} | obs '{OBS}'")
        ds = physfit_stream_datasets(BANKDIR, grids, nom, log, obs=obs)
        nbins = sum(d["nbin"] for d in ds); log(f"{len(ds)} datasets, {nbins} bins")
        edges_by = {d["key"]: d["edges"] for d in ds}; dkeys = [d["key"] for d in ds]
        J, row0 = FE.bank_jacobian_chunked(jvp_wf, th0, BANKDIR,
                                           lambda Bc: _physfit_bin_ds(Bc, edges_by, dkeys),
                                           [d["nbin"] for d in ds], NPAR, log=log, names=PNAMES)
    else:                                          # whole-bank path (unchanged): load once, one jvp per knob
        B = BP.load_bank(BANKDIR); JB = BR.to_jax(B)
        w0 = np.asarray(BR.weight_jit(JB, nom, grids))
        log(f"bank {len(w0)} events | {NPAR} knobs | {N_BINS} bins/obs | syst {SYST:.0%} | obs set '{OBS}'")
        ds = build_physfit_datasets(B, w0, log, obs=obs)
        nbins = sum(d["nbin"] for d in ds); log(f"{len(ds)} datasets, {nbins} bins")
        J, row0 = FE.bank_jacobian(jvp_wf, th0, JB, ds, NPAR, log=log, names=PNAMES)
        del B, JB
    sigma = np.concatenate([d["sigma"] for d in ds])

    if T2K_CH:                                    # CC1pi+ CH: add the free-H GRADIENT (central already CH)
        log("CH mode: adding reweightable free-H CC1pi+ gradient from nu_T2K_H")
        cH = cc1pi_freeH_jacobian(ds, row0, J, grids, nom, log)
        for d in ds:                              # consistency: bank free-H central vs the frozen offset
            if d["key"] in cH:
                log(f"  [check {d['key']}] free-H central sum: bank={cH[d['key']].sum():.4e} "
                    f"frozen_offset={np.asarray(d['offset']).sum():.4e}")

    # ---- Gate I: Asimov Fisher + priors ----------------------------------------------------------- #
    F, V, sig_post, shrink, _reach = FE.gate1(J, sigma, PRIOR)
    verdict = np.where(shrink < 0.5, "FIT", "freeze")
    order = np.argsort(shrink)
    print(f"\n==== GATE I (Asimov Fisher, {nbins} bins, syst {SYST:.0%}, priors: 20% mult / natural) ====")
    print(f"{'knob':>16} {'prior':>7} {'sig_post':>9} {'shrink':>7}  verdict")
    for k in order:
        print(f"{PNAMES[k]:>16} {PRIOR[k]:7.2f} {sig_post[k]:9.3f} {shrink[k]:7.2f}  {verdict[k]}")
    nfit = int((shrink < 0.5).sum())
    print(f"\n{nfit}/{NPAR} knobs pass Gate I (shrinkage < 0.5)")

    # ---- degeneracy structure: eigen-spectrum of the PRIOR-SCALED Fisher (dimensionless, so knobs in
    # natural units are commensurate with the multiplicative ones). Small eigenvalue = flat direction. --
    Fs = F * PRIOR[:, None] * PRIOR[None, :]
    evals, evecs = np.linalg.eigh(Fs)
    corr = V / np.outer(sig_post, sig_post)                     # posterior correlation matrix
    print(f"\n---- Fisher eigen-spectrum (prior-scaled; lambda < 1 = prior-dominated) ----")
    for i in np.argsort(evals)[::-1]:
        top = np.argsort(np.abs(evecs[:, i]))[::-1][:3]
        comp = ", ".join(f"{evecs[t, i]:+.2f} {PNAMES[t]}" for t in top)
        print(f"  lambda {evals[i]:10.3g}   {comp}")

    os.makedirs("output/altgen", exist_ok=True)
    np.savez(f"output/altgen/{LABEL}.npz", J=J, sigma=sigma, F=F, V=V, prior=PRIOR,
             sig_post=sig_post, shrink=shrink, pnames=PNAMES, nbins=nbins,
             fisher_evals=evals, fisher_evecs=evecs, corr=corr, obs_set=OBS,
             row0=row0, dsnames=[d["name"] for d in ds], dskeys=[d["key"] for d in ds],
             **{f"{d['key']}_edges": d["edges"] for d in ds},
             **{f"{d['key']}_central": d["data"] for d in ds},
             **{f"{d['key']}_sigma": d["sigma"] for d in ds})
    log(f"[out] output/altgen/{LABEL}.npz")
    log("done")


if __name__ == "__main__":
    main()
