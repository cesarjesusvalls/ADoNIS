"""Signal selection + TKI/STV observables on the CURRENT bank schemas, driven by a NuSignalDef.

Two inputs, ONE selection vocabulary:
  * ADoNIS   -> `bank_signal(bank_dir, sd)`   : a paper_banks bank dir (k_lep + ragged fs_*, weight w0;
                loaded by bank_plot.load_bank, which already divides w0 by n_chunks -> absolute nb).
  * ACHILLES -> `oracle_signal(oracle_npz, sd)`: a probe-agnostic fs_rich oracle npz
                (lep / prot_p4 / pi_p4 / pi_pid / n_other_meson, weight w * weight_to_nb -> absolute nb).

The NuSignalDef selects the topology (pion_id "none" = CC0pi, "pip" = CC1pi) and the acceptance windows
(mu_win + cos_mu|cth, p_win + cth, pi_win + cth).  The SAME cuts and the SAME validated STV formulas
(bank_plot.pN_1pi/dptt_1pi/dpt_1pi/dat_1pi -- with the pion 4-vector set to zero, the CC1pi formulas
reduce EXACTLY to CC0pi) run on both sides.  This replaces the retired engine-bank
select_signal/select_reference (P-A); it generalizes the T2K/MINERvA/MicroBooNE scratch drivers.
"""
import glob
import json

import numpy as np

from adonis.reweight import bank_plot as BP
from adonis.constants import (PDG_MESONS, mp as _MP, mN as _MN, me as _ME, mpip as _MPIP,
                              MASS_PDG_MUON as _MMU)

_MESONS = list(PDG_MESONS)   # np.isin needs a list/tuple -- a frozenset silently matches NOTHING


def _cos(p4, mom):
    return np.where(mom > 0, p4[:, 3] / np.maximum(mom, 1e-9), -2.0)


def _obs(mu, lead, pip, knu=None):
    """TKI/STV observable dict from raw (mu, lead, pip) lab 4-vectors (pip = zeros for CC0pi).
    Reuses the validated bank_plot raw formulas; CC0pi is exactly CC1pi with the pion set to zero."""
    pmu = np.linalg.norm(mu[:, 1:], axis=1)
    pl = np.linalg.norm(lead[:, 1:], axis=1)
    ppi = np.linalg.norm(pip[:, 1:], axis=1)
    cmu, clp, cpi = _cos(mu, pmu), _cos(lead, pl), _cos(pip, ppi)
    return {"dpt": np.asarray(BP.dpt_1pi(mu, lead, pip)),          # |pT^mu + pT^p (+ pT^pi)| [MeV]
            "dalphat": np.asarray(BP.dat_1pi(mu, lead, pip)),      # delta_alphaT [rad]
            "dphit": np.asarray(BP.dphit_1pi(mu, lead, pip)),      # delta_phiT [rad] (deflecting angle)
            "pn": np.asarray(BP.pN_1pi(mu, lead, pip)),            # inferred nucleon |p| (carbon reco) [MeV]
            "dptt": np.asarray(BP.dptt_1pi(mu, lead, pip)),        # double-transverse imbalance [MeV] (CC1pi)
            "pmu": pmu, "cos_mu": cmu, "th_mu": np.degrees(np.arccos(np.clip(cmu, -1.0, 1.0))),
            "th_mu_rad": np.arccos(np.clip(cmu, -1.0, 1.0)),       # same angle in RADIANS

            "pt": np.sqrt(mu[:, 1] ** 2 + mu[:, 2] ** 2), "pz": mu[:, 3],   # MINERvA qelike muon pT/p||
            "lp_p": pl, "cos_lp": clp, "th_lp": np.degrees(np.arccos(np.clip(clp, -1.0, 1.0))),
            "ppi": ppi, "cos_pi": cpi,
            # MINERvA CC1pi+ variables.  T_pi is the pion KINETIC energy; Q2 and W_exp follow the paper's
            # nucleon-at-rest reconstruction (arXiv:2605.24224 Eqs. 1-3) evaluated with the TRUE Enu --
            # equivalent to their E_had = Enu - Emu, and free of any visible-energy convention.
            "tpi": np.maximum(pip[:, 0] - _MPIP, 0.0),
            **_q2_wexp(mu, pmu, cmu, knu)}


def _q2_wexp(mu, pmu, cmu, knu):
    """Q2 [GeV^2] and W_exp [MeV], nucleon at rest (arXiv:2605.24224):
         Q2     = 2 Enu (Emu - |pmu| cos_mu) - mmu^2
         W_exp^2 = mN^2 - Q2 + 2 mN (Enu - Emu)
    knu = true neutrino 4-vector; None -> NaN (keys always present so every path returns one schema)."""
    if knu is None:
        nan = np.full(len(mu), np.nan)
        return {"q2": nan, "w_exp": nan}
    Enu = np.asarray(knu)[:, 0].astype(np.float64); Emu = mu[:, 0]
    q2 = 2.0 * Enu * (Emu - pmu * cmu) - _MMU ** 2                      # MeV^2
    w2 = _MN ** 2 - q2 + 2.0 * _MN * (Enu - Emu)
    return {"q2": q2 * 1e-6, "w_exp": np.sqrt(np.maximum(w2, 0.0))}


def _mu_pass(pmu, cmu, sd):
    cut = sd.cos_mu if sd.cos_mu is not None else sd.cth       # CC0pi backward/forward mu cut, else cth
    return (pmu >= sd.mu_win[0]) & (pmu <= sd.mu_win[1]) & (cmu > cut)


def _finish(obs, sel, w, chan):
    out = {k: v[sel] for k, v in obs.items()}
    out["w"] = np.asarray(w)[sel]
    out["chan"] = np.asarray(chan)[sel]          # 0 = QE, 1 = RES (for the reference QE/RES breakdown)
    return out


# --------------------------------------------------------------------------- streaming (bounded memory)
# The ADoNIS banks are 20M events across ~240 chunk_*.npz; a signal cut keeps ~0.1-1% of them.  Loading
# the whole concatenated bank just to select that sliver OOMs the big Ar/uBooNE banks (needs ~128G).  So
# every bank-side reducer STREAMS: load one chunk, keep only the passing events, free the chunk.  Peak
# memory = one chunk + the (tiny) accumulated selection -- independent of bank size.  Exactly reproduces
# load-all-then-select: load_bank_chunk divides w0 by the manifest n_chunks (as load_bank does), and chunk
# + within-chunk order is preserved, so the histograms/chi2 are bit-identical.
def _merge_sel(parts):
    parts = [p for p in parts if len(p["w"])]
    if not parts:
        raise ValueError("selection returned no events on any chunk")
    keys = list(parts[0])
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def _stream_select(bank_dir, loader, select_fn, sd):
    nch = BP.bank_nchunks(bank_dir)
    parts = []
    for f in sorted(glob.glob(f"{bank_dir}/chunk_*.npz")):
        B = loader(f, nch)
        parts.append(select_fn(B, sd))
        del B                                    # free the chunk before the next one loads
    return _merge_sel(parts)


# --------------------------------------------------------------------------- ADoNIS paper_banks side
def _cc_full(B, sd):
    """Full-length (sel mask, obs dict, w0, chan) for ONE CC bank/chunk -- the pre-compaction body shared
    by bank_signal (which _finish-compacts it for plotting) and the Gate-I jacobian (which bins the
    per-event DIFFERENTIATED weight over the SAME mask/edges).  sel/obs/w0/chan are all bank-length."""
    mu = B["k_lep"].astype(np.float64)
    pmu = np.linalg.norm(mu[:, 1:], axis=1); cmu = _cos(mu, pmu)
    if sd.pion_id == "none":                                   # ---- CC0pi (Np) / CC-inclusive muon box ----
        n_meson = BP._event_sum(B, np.isin(B["fs_pid"], _MESONS).astype(float))
        lead, hasp = BP.leading_proton(B)                      # global leading proton (NUISANCE def)
        pip = np.zeros_like(mu)
        sel = (n_meson == 0) & _mu_pass(pmu, cmu, sd)
        if sd.require_proton:                                  # NUISANCE CC0pi-Np: leading proton in window
            pl = np.linalg.norm(lead[:, 1:], axis=1); cl = _cos(lead, pl)
            sel = sel & hasp & (pl > sd.p_win[0]) & (pl < sd.p_win[1]) & (cl > sd.cth)
            if sd.proton_count == "eq1":                       # exactly ONE proton in acceptance (CC1p0pi)
                pid = B["fs_pid"]; p4p = B["fs_p4"]; pmp = np.linalg.norm(p4p[:, 1:], axis=1)
                inacc = (pid == 2212) & (pmp > sd.p_win[0]) & (pmp < sd.p_win[1]) & (_cos(p4p, pmp) > sd.cth)
                sel = sel & (BP._event_sum(B, inacc.astype(float)).astype(int) == 1)
        if sd.pt_hi is not None:                               # MINERvA qelike muon pT cap (hadron-inclusive)
            sel = sel & (np.sqrt(mu[:, 1] ** 2 + mu[:, 2] ** 2) <= sd.pt_hi)
        if sd.pz_win is not None:                              # + muon p|| window
            sel = sel & (mu[:, 3] >= sd.pz_win[0]) & (mu[:, 3] <= sd.pz_win[1])
    else:                                                      # ---- CC1pi ----
        npip, npi0, npim = BP.pion_counts(B); pip = BP.single_pip(B)
        if sd.pion_id == "anypi":
            # "anypi" USED to validate and then do nothing -- selection never branched on it, so it
            # was byte-identical to "pip".  A config key that silently does nothing is the same class
            # of defect as a field name that lies.  Implemented: exactly one pion of ANY charge, and
            # the signal pion is whichever one it is.
            pip = pip + BP.single_pi0(B) + BP._single_pion(B, -211)
        elif sd.pion_id != "pip":
            raise ValueError(f"pion_id={sd.pion_id!r} has no selection branch; for NC1pi0 use "
                             "bank_signal_nc / oracle_signal_nc, which are a separate path")
        lead, hasp = BP.leading_proton_window(B, sd.p_win[0], sd.p_win[1], cth=sd.cth)
        ppi = np.linalg.norm(pip[:, 1:], axis=1); cpi = _cos(pip, ppi)
        npi_tot = npip + npi0 + npim
        one_pion = (npi_tot == 1) if sd.pion_id == "anypi" else \
                   ((npip == 1) & (npi0 == 0) & (npim == 0))
        sel = one_pion & _mu_pass(pmu, cmu, sd)
        if sd.require_proton:                                  # T2K CC1pi+Np: leading proton in window
            sel = sel & hasp
        if sd.tpi_win is not None:                             # MINERvA: pion KINETIC-energy window
            tpi = np.maximum(pip[:, 0] - _MPIP, 0.0)
            sel = sel & (tpi >= sd.tpi_win[0]) & (tpi < sd.tpi_win[1])
        else:                                                  # T2K: pion momentum window + forward cut
            sel = sel & (ppi >= sd.pi_win[0]) & (ppi < sd.pi_win[1]) & (cpi > sd.cth)
        if sd.veto_other_mesons:                               # "no other mesons" (eta, K, ...)
            n_heavy = BP._event_sum(B, (np.isin(B["fs_pid"], _MESONS)
                                        & ~np.isin(B["fs_pid"], [211, -211, 111])).astype(float))
            sel = sel & (n_heavy == 0)
        if sd.w_exp_max is not None:                           # W_exp < 1.4 GeV (Delta region)
            sel = sel & (_q2_wexp(mu, pmu, cmu, B["k_nu"])["w_exp"] < sd.w_exp_max)
    return (sel, _obs(mu, lead, pip, B["k_nu"] if "k_nu" in B else None),
            np.asarray(B["w0"]), np.asarray(B["channel"]))     # chan: 0 QE, 1 RES


def _select_cc(B, sd):
    """CC selection compacted to the passing events -- the streaming bank_signal reducer (unchanged output)."""
    sel, obs, w0, chan = _cc_full(B, sd)
    return _finish(obs, sel, w0, chan)


def _ele_full_bank(B, sd):
    """Full-length (mask, obs, w0, chan) for the (e,e') gradient on a STANDARD bank chunk (BP.load_bank_chunk,
    which carries omega/theta AND the EM reweight records the jvp needs -- unlike the plot-side
    _load_ele_chunk).  mask = scattered electron inside the theta window & non-zero weight (the generation
    acceptance is already baked into the bank, so this reproduces the old physfit good=w0>0)."""
    theta = np.asarray(B["theta"], float)
    Ee = float(sd.beam_energy) - np.asarray(B["omega"], float)
    w0 = np.asarray(B["w0"], float)
    mask = (theta >= sd.e_theta_win[0]) & (theta <= sd.e_theta_win[1]) & (w0 > 0)
    if sd.e_min is not None:
        mask = mask & (Ee >= sd.e_min)
    om = np.asarray(B["omega"], float)
    if getattr(sd, "omega_win", None) is not None:                # range is a SELECTION, not an overflow bin
        mask = mask & (om >= sd.omega_win[0]) & (om <= sd.omega_win[1])
    return mask, {"omega": om}, w0, np.asarray(B["channel"])


def select_full(B, sd):
    """Full-length selection dispatch (mask + obs dict + w0 + chan) for the Gate-I jacobian, which bins the
    per-event DIFFERENTIATED weight over the SAME mask/edges bank_signal plots.  Dispatches on the signal
    type: EleBeamSignalDef -> (e,e') electron; NuSignalDef -> CC.  (NC gradient is a later sample.)"""
    if hasattr(sd, "e_theta_win"):                            # EleBeamSignalDef (duck-typed to avoid a cycle)
        return _ele_full_bank(B, sd)
    return _cc_full(B, sd)


def bank_signal(bank_dir, sd):
    """CC selection over an ADoNIS bank -- STREAMED chunk-by-chunk (bounded memory, see _stream_select)."""
    return _stream_select(bank_dir, BP.load_bank_chunk, _select_cc, sd)


# --------------------------------------------------------------------------- N_selected compact bank
_EIDX_FAM = ("f_p_eidx", "f_n_eidx", "ks_eidx")     # per-slot event indices to shift into global numbering


def _concat_compact(parts):
    """Concatenate per-chunk compact banks (bank_plot.filter_events output) into ONE, shifting every ragged
    event index into the global numbering (like load_bank does for f_*_eidx, extended to ks_eidx) and
    rebuilding fs_off/_eidx.  w0 is already per-chunk normalized, so a plain concatenation sums to the
    (subsample) cross section."""
    import numpy as _np
    parts = [p for p in parts if len(p["w0"])]
    if not parts:
        raise ValueError("select_bank: no selected events on any chunk")
    keys = [k for k in parts[0] if k not in ("fs_off", "_eidx", "n_chunks", "labels")]
    acc = {k: [] for k in keys}; fs_off = [_np.array([0], _np.int64)]; ev_off = 0
    for p in parts:
        for k in keys:
            v = _np.asarray(p[k])
            acc[k].append(v.astype(_np.int64) + ev_off if k in _EIDX_FAM else v)
        fs_off.append(fs_off[-1][-1] + _np.asarray(p["fs_off"])[1:])
        ev_off += len(p["w0"])
    out = {k: _np.concatenate(acc[k]) for k in keys}
    out["fs_off"] = _np.concatenate(fs_off); out["n_chunks"] = parts[0].get("n_chunks", 1)
    n = len(out["w0"]); out["_eidx"] = _np.repeat(_np.arange(n), _np.diff(out["fs_off"]))
    return out


def select_bank(bank_dir, sd, max_chunks=None, cap=None):
    """A compact full-record bank of ONLY the signal events (N_selected), built by STREAMING the full bank
    and filter_events-ing each chunk to `sd`.  Peak memory = one chunk + the accumulated signal (never the
    whole bank); bank_weight / the observables on the result reproduce the full bank restricted to the
    signal.  This is what lets the fit (sec4) cache N_selected instead of N_total.

    `cap` (fit-side subsample): keep EXACTLY `cap` selected events.  Stream chunks; when a chunk would push
    the total past cap, keep only its first (cap - accumulated) selected events and stop.  w0 is normalized
    by the EFFECTIVE chunk count = (full chunks kept) + (fraction of the last chunk's SELECTED events kept).
    That is an exact, UNBIASED cross section: each chunk's selected-w0 sum is an iid estimate of
    sigma_selected (different seeds), and a random fraction f of a chunk's selected events sums to ~f*sigma,
    so dividing by K+f recovers sigma.  No partial-chunk bias.  cap=None -> the full signal (loaded chunk
    count).  (Events aren't physically ordered within a chunk, so the first-N is a fine random subsample.)"""
    import glob
    files = sorted(glob.glob(f"{bank_dir}/chunk_*.npz"))
    if max_chunks:
        files = files[:max_chunks]
    parts = []; nsel = 0; eff = 0.0
    for f in files:
        B = BP.load_bank_chunk(f, 1)                       # raw w0 (undivided); divide by the eff. count below
        Bc = BP.filter_events(B, select_full(B, sd)[0]); m = len(Bc["w0"]); del B
        if cap is not None and nsel + m > cap:
            take = cap - nsel
            keep = np.zeros(m, bool); keep[:take] = True   # first `take` selected events of this chunk
            parts.append(BP.filter_events(Bc, keep)); eff += take / m; nsel += take
            break
        parts.append(Bc); nsel += m; eff += 1.0
    out = _concat_compact(parts)
    out["w0"] = np.asarray(out["w0"]) / eff                # unbiased sigma from the effective chunk count
    return out


# --------------------------------------------------------------------------- ACHILLES fs_rich side
def oracle_signal(oracle_npz, sd):
    d = np.load(oracle_npz, allow_pickle=True)
    w = np.asarray(d["w"], float) * float(d["weight_to_nb"])
    mu = np.asarray(d["lep"], float)                          # outgoing lepton (mu OR e-)
    pmu = np.linalg.norm(mu[:, 1:], axis=1); cmu = _cos(mu, pmu)
    prot = np.asarray(d["prot_p4"], float)                    # (n, M, 4) padded, |p|-sorted
    pm = np.linalg.norm(prot[:, :, 1:], axis=2)
    pipid = np.asarray(d["pi_pid"]); pi4 = np.asarray(d["pi_p4"], float)
    n_other = np.asarray(d["n_other_meson"]) if "n_other_meson" in d.files else np.zeros(len(mu))
    proc = np.asarray(d["proc"]) if "proc" in d.files else np.full(len(mu), 200)
    chan = (proc != 200).astype(int)                          # ACHILLES proc: 200 = QE, 401/402 = RES
    idx = np.arange(len(mu))
    if sd.pion_id == "none":                                  # ---- CC0pi ----
        n_meson = (pipid != 0).sum(1) + n_other.astype(int)
        j = pm.argmax(1); lead = prot[idx, j]; pl = pm.max(1); cl = _cos(lead, pl)
        pip = np.zeros_like(mu)
        sel = ((n_meson == 0) & _mu_pass(pmu, cmu, sd)
               & (pl > sd.p_win[0]) & (pl < sd.p_win[1]) & (cl > sd.cth))
        if sd.proton_count == "eq1":                          # exactly ONE proton in acceptance (CC1p0pi)
            czp = _cos(prot.reshape(-1, 4), pm.reshape(-1)).reshape(pm.shape)
            nprot = ((pm > sd.p_win[0]) & (pm < sd.p_win[1]) & (czp > sd.cth)).sum(1)
            sel = sel & (nprot == 1)
    else:                                                     # ---- CC1pi+ ----
        is_pip = (pipid == 211)
        npip = is_pip.sum(1); npi0 = (pipid == 111).sum(1); npim = (pipid == -211).sum(1)
        ip = np.argmax(is_pip, axis=1); pip = pi4[idx, ip]
        ppi = np.linalg.norm(pip[:, 1:], axis=1); cpi = _cos(pip, ppi)
        cz = _cos(prot.reshape(-1, 4), pm.reshape(-1)).reshape(pm.shape)
        acc = (pm >= sd.p_win[0]) & (pm < sd.p_win[1]) & (cz > sd.cth)
        key = np.where(acc, pm, -1.0); j = key.argmax(1); lead = prot[idx, j]
        haslead = key[idx, j] > 0
        sel = ((npip == 1) & (npi0 == 0) & (npim == 0) & haslead & _mu_pass(pmu, cmu, sd)
               & (ppi >= sd.pi_win[0]) & (ppi < sd.pi_win[1]) & (cpi > sd.cth))
    return _finish(_obs(mu, lead, pip), sel, w, chan)


# =============================================================== NC1pi0: a PARALLEL selection path
# Deliberately NOT a flag threaded through _obs.  Every observable it computes -- dpt, dalphat,
# dphit, pn, dptt -- takes the outgoing lepton as a REQUIRED argument and is physically undefined
# without it.  For NC the lepton is an invisible neutrino, so threading a "lepton optional" flag
# would produce observables that are silently meaningless rather than absent, which is worse.
#
# The NC observables are the ones the paper's figures 11 and 12 actually plot: p_pi0, cos theta_pi0,
# and the proton multiplicity ("Xp").

def _obs_nc(pi0, protons_lead, has_p):
    """Observable dict for an NC1pi0 event.  No lepton anywhere."""
    ppi = np.linalg.norm(pi0[:, 1:], axis=1)
    cpi = _cos(pi0, ppi)
    pl = np.linalg.norm(protons_lead[:, 1:], axis=1)
    return {"p_pi0": ppi, "cos_pi0": cpi,
            "th_pi0": np.degrees(np.arccos(np.clip(cpi, -1.0, 1.0))),
            "lp_p": np.where(has_p, pl, 0.0), "cos_lp": _cos(protons_lead, pl),
            "has_proton": has_p.astype(float)}


def _select_nc(B, sd):
    """NC1pi0(Xp) selection body on ONE chunk's bank dict (streamed by bank_signal_nc): EXACTLY one pi0,
    no charged pions, no non-pion mesons."""
    npip, npi0, npim = BP.pion_counts(B)
    pi0 = BP.single_pi0(B)
    lead, hasp = BP.leading_proton(B)                       # global leading proton (Xp: 0 or more)
    ppi = np.linalg.norm(pi0[:, 1:], axis=1); cpi = _cos(pi0, ppi)
    # non-pion mesons (eta, K, ...) counted directly off the final state -- NOT via n_other_meson,
    # which counts pi0 as an "other meson" and would veto 100% of this signal.
    n_heavy = BP._event_sum(B, (np.isin(B["fs_pid"], _MESONS)
                                & ~np.isin(B["fs_pid"], (211, 111, -211))).astype(float))
    sel = ((npi0 == 1) & (npip == 0) & (npim == 0) & (n_heavy == 0)
           & (ppi >= sd.pi_win[0]) & (ppi < sd.pi_win[1]) & (cpi > sd.cth))
    if sd.proton_count == "eq1":
        sel = sel & hasp
    return _finish(_obs_nc(pi0, lead, hasp), sel, B["w0"], B["channel"])


def bank_signal_nc(bank_dir, sd):
    """NC1pi0(Xp) over an ADoNIS bank -- STREAMED chunk-by-chunk (bounded memory)."""
    return _stream_select(bank_dir, BP.load_bank_chunk, _select_nc, sd)


def oracle_signal_nc(oracle_npz, sd):
    """NC1pi0(Xp) on an ACHILLES fs_rich oracle -- the SAME cuts as bank_signal_nc.

    Uses the n_nonpion_meson field (true non-pion mesons only), never n_other_meson: the latter
    counts pi0 and pi- as "other mesons", which is harmless for CC0pi (both terms are zero on signal)
    and fatal here.  A file predating that field raises rather than silently mis-vetoing."""
    d = np.load(oracle_npz, allow_pickle=True)
    if "n_nonpion_meson" not in d.files:
        raise KeyError(f"{oracle_npz} has no 'n_nonpion_meson' -- re-extract it.  Falling back to "
                       "'n_other_meson' would veto the pi0 signal, since that field counts pi0 as an "
                       "'other meson'.")
    w = np.asarray(d["w"], float) * float(d["weight_to_nb"])
    pipid = np.asarray(d["pi_pid"]); pi4 = np.asarray(d["pi_p4"], float)
    n_heavy = np.asarray(d["n_nonpion_meson"]).astype(int)
    prot = np.asarray(d["prot_p4"], float)
    pm = np.linalg.norm(prot[:, :, 1:], axis=2)
    idx = np.arange(len(w))
    is_pi0 = (pipid == 111)
    npi0 = is_pi0.sum(1); npip = (pipid == 211).sum(1); npim = (pipid == -211).sum(1)
    ip = np.argmax(is_pi0, axis=1); pi0 = pi4[idx, ip]
    j = pm.argmax(1); lead = prot[idx, j]; hasp = pm.max(1) > 0
    ppi = np.linalg.norm(pi0[:, 1:], axis=1); cpi = _cos(pi0, ppi)
    proc = np.asarray(d["proc"]) if "proc" in d.files else np.full(len(w), 451)
    chan = np.isin(proc, (451, 452)).astype(int)     # NC: 250/251 = QE, 451/452 = RES (measured)
    sel = ((npi0 == 1) & (npip == 0) & (npim == 0) & (n_heavy == 0)
           & (ppi >= sd.pi_win[0]) & (ppi < sd.pi_win[1]) & (cpi > sd.cth))
    if sd.proton_count == "eq1":
        sel = sel & hasp
    return _finish(_obs_nc(pi0, lead, hasp), sel, w, chan)


# =============================================================== electron (e,e') beam: a PEER of the nu path
# The sibling of bank_signal / oracle_signal for the electron-scattering figures, driven by an
# EleBeamNuSignalDef instead of a NuNuSignalDef.  Same contract: one reducer per input side, each returning a
# per-event dict of the (e,e') observables (omega, E_QE, E_cal, P_T) + weight w + QE/RES channel + the
# topology counts (npi, nprot) the figures slice on (0pi for E_QE, 1p0pi for E_cal/P_T).  The common
# electron acceptance (theta window + optional E_e floor) is applied here; the per-observable 0pi/1p0pi
# masks are applied by the caller, since one figure histograms several topologies off one reduction.
#
# The (e,e') banks differ from the neutrino paper_banks on disk (weight field `c` not `w0`; the inclusive
# bank carries no outgoing-lepton 4-vector, since omega+theta fully fix the scattered electron), so this
# path uses its own small loader rather than bank_plot.load_bank.

def _load_ele_bank(bank_dir):
    """Concatenate an (e,e') bank's chunks -> per-event {omega, theta, k_lep, w0, channel, n_pi_out} +
    ragged fs_pid/fs_p4/fs_off (+ _eidx).  w0 = c / n_chunks (absolute nb).  k_lep is zeros when absent
    (inclusive bank: only omega/theta are needed there)."""
    man = json.load(open(f"{bank_dir}/manifest.json")); nch = man["n_chunks"]
    files = sorted(glob.glob(f"{bank_dir}/chunk_*.npz"))
    per = {k: [] for k in ("omega", "theta", "channel", "n_pi_out", "k_lep", "w0")}
    fs_pid, fs_p4, offs = [], [], [np.array([0], np.int64)]
    for f in files:
        d = np.load(f); n = len(d["c"])
        per["omega"].append(np.asarray(d["omega"], float))
        per["theta"].append(np.asarray(d["theta"], float))
        per["channel"].append(np.asarray(d["channel"]))
        per["n_pi_out"].append(np.asarray(d["n_pi_out"]) if "n_pi_out" in d.files else np.zeros(n, int))
        lep = next((o for o in ("k_lep", "k_e", "k_mu") if o in d.files), None)
        per["k_lep"].append(np.asarray(d[lep], float) if lep else np.zeros((n, 4)))
        per["w0"].append(np.asarray(d["c"], float) / nch)
        fs_pid.append(d["fs_pid"]); fs_p4.append(d["fs_p4"])
        offs.append(offs[-1][-1] + d["fs_off"][1:])
    B = {k: np.concatenate(v) for k, v in per.items()}
    B["fs_pid"] = np.concatenate(fs_pid); B["fs_p4"] = np.concatenate(fs_p4); B["fs_off"] = np.concatenate(offs)
    B["_eidx"] = np.repeat(np.arange(len(B["w0"])), np.diff(B["fs_off"]))
    return B


def _ele_obs(Ee, cth, klep, lead, eps):
    """(e,e') reconstructed observables from the scattered electron (Ee, cth, klep) + the leading proton.
    omega is added by the caller (it is E_beam - Ee, and E_beam differs by side)."""
    pe = np.sqrt(np.maximum(Ee ** 2 - _ME ** 2, 0.0))
    E_QE = (2 * _MN * eps + 2 * _MN * Ee - _ME ** 2) / (2 * (_MN - Ee + pe * cth))
    E_cal = Ee + (lead[:, 0] - _MP) + eps                     # T_p = E_p - m_p
    P_T = np.sqrt((klep[:, 1] + lead[:, 1]) ** 2 + (klep[:, 2] + lead[:, 2]) ** 2)
    return {"E_QE": E_QE, "E_cal": E_cal, "P_T": P_T}


def _ele_lead_bank(B, sd):
    """Leading in-acceptance proton (|p|>p_min, theta in p_theta_win) per event + count, off the ragged
    final state.  Zeros/0 when the def has no proton window (inclusive figure)."""
    n = len(B["w0"]); lead = np.zeros((n, 4)); nprot = np.zeros(n)
    if sd.p_min is None or sd.p_theta_win is None:
        return lead, nprot.astype(int)
    pid = B["fs_pid"]; p4 = B["fs_p4"]; eidx = B["_eidx"]
    mom = np.linalg.norm(p4[:, 1:], axis=1)
    cz = np.where(mom > 0, p4[:, 3] / np.maximum(mom, 1e-9), -2.0)
    th = np.degrees(np.arccos(np.clip(cz, -1, 1)))
    acc = (pid == 2212) & (mom > sd.p_min) & (th >= sd.p_theta_win[0]) & (th <= sd.p_theta_win[1])
    np.add.at(nprot, eidx[acc], 1.0)
    key = np.where(acc, mom, -1.0); mx = np.full(n, -1.0); np.maximum.at(mx, eidx, key)
    islead = acc & (key == mx[eidx]) & (mom > 0)
    lead[eidx[islead]] = p4[islead]
    return lead, nprot.astype(int)


def _ele_finish(obs, elec, w, chan, npi, nprot):
    out = {k: v[elec] for k, v in obs.items()}
    out["w"] = np.asarray(w)[elec]; out["chan"] = np.asarray(chan)[elec]
    out["npi"] = np.asarray(npi)[elec]; out["nprot"] = np.asarray(nprot)[elec]
    return out


def _load_ele_chunk(f, nch):
    """ONE (e,e') bank chunk as the dict _select_ele expects (w0 = c/nch).  Single-chunk peer of
    _load_ele_bank, for the streaming (bounded-memory) selection."""
    d = np.load(f); n = len(d["c"])
    lep = next((o for o in ("k_lep", "k_e", "k_mu") if o in d.files), None)
    B = {"omega": np.asarray(d["omega"], float), "theta": np.asarray(d["theta"], float),
         "channel": np.asarray(d["channel"]),
         "n_pi_out": np.asarray(d["n_pi_out"]) if "n_pi_out" in d.files else np.zeros(n, int),
         "k_lep": np.asarray(d[lep], float) if lep else np.zeros((n, 4)),
         "w0": np.asarray(d["c"], float) / nch,
         "fs_pid": d["fs_pid"], "fs_p4": d["fs_p4"], "fs_off": d["fs_off"]}
    B["_eidx"] = np.repeat(np.arange(n), np.diff(B["fs_off"]))
    return B


def _select_ele(B, sd):
    """(e,e') selection body on ONE chunk's bank dict (streamed by ele_signal)."""
    theta = np.asarray(B["theta"], float)                     # degrees
    Ee = float(sd.beam_energy) - np.asarray(B["omega"], float)
    cth = np.cos(np.radians(theta)); klep = B["k_lep"].astype(np.float64)
    elec = (theta >= sd.e_theta_win[0]) & (theta <= sd.e_theta_win[1])
    if sd.e_min is not None:
        elec = elec & (Ee >= sd.e_min)
    om = np.asarray(B["omega"], float)
    if getattr(sd, "omega_win", None) is not None:                # keep in step with _ele_full_bank
        elec = elec & (om >= sd.omega_win[0]) & (om <= sd.omega_win[1])
    lead, nprot = _ele_lead_bank(B, sd)
    obs = {"omega": om, **_ele_obs(Ee, cth, klep, lead, sd.removal_energy)}
    return _ele_finish(obs, elec, B["w0"], B["channel"], B["n_pi_out"], nprot)


def ele_signal(bank_dir, sd):
    """(e,e') selection over an ADoNIS bank -- STREAMED chunk-by-chunk (bounded memory)."""
    return _stream_select(bank_dir, _load_ele_chunk, _select_ele, sd)


def ele_oracle_signal(oracle_npz, sd):
    """(e,e') on an ACHILLES fs_rich oracle -- the SAME acceptance + reconstruction as ele_signal."""
    d = np.load(oracle_npz, allow_pickle=True)
    w = np.asarray(d["w"], float) * float(d["weight_to_nb"])
    lep = np.asarray(d["lep"], float); Ee = lep[:, 0]
    pe = np.linalg.norm(lep[:, 1:], axis=1); cth = np.where(pe > 0, lep[:, 3] / np.maximum(pe, 1e-9), -2.0)
    theta = np.degrees(np.arccos(np.clip(cth, -1, 1)))
    elec = (theta >= sd.e_theta_win[0]) & (theta <= sd.e_theta_win[1])
    if sd.e_min is not None:
        elec = elec & (Ee >= sd.e_min)
    if getattr(sd, "omega_win", None) is not None:                # ORACLE side: same cut, else ADoNIS/ACHILLES
        _om = float(sd.beam_energy) - Ee                          # comparisons would be over different ranges
        elec = elec & (_om >= sd.omega_win[0]) & (_om <= sd.omega_win[1])
    n = len(w)
    if "prot_p4" in d.files and sd.p_min is not None and sd.p_theta_win is not None:
        prot = np.asarray(d["prot_p4"], float); pm = np.linalg.norm(prot[:, :, 1:], axis=2)
        pcz = np.where(pm > 0, prot[:, :, 3] / np.maximum(pm, 1e-9), -2.0)
        thp = np.degrees(np.arccos(np.clip(pcz, -1, 1)))
        acc = (pm > sd.p_min) & (thp >= sd.p_theta_win[0]) & (thp <= sd.p_theta_win[1])
        nprot = acc.sum(1)
        key = np.where(acc, pm, -1.0); j = key.argmax(1); lead = prot[np.arange(n), j]
    else:
        nprot = np.zeros(n, int); lead = np.zeros((n, 4))
    npi = (np.asarray(d["pi_pid"]) != 0).sum(1) if "pi_pid" in d.files else np.zeros(n, int)
    obs = {"omega": float(sd.beam_energy) - Ee, **_ele_obs(Ee, cth, lep, lead, sd.removal_energy)}
    return _ele_finish(obs, elec, w, np.zeros(n, int), npi, nprot)   # chan set by the caller per qe/res file
