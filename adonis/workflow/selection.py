"""Signal selection + TKI/STV observables on the CURRENT bank schemas, driven by a SignalDef.

Two inputs, ONE selection vocabulary:
  * ADoNIS   -> `bank_signal(bank_dir, sd)`   : a paper_banks bank dir (k_lep + ragged fs_*, weight w0;
                loaded by bank_plot.load_bank, which already divides w0 by n_chunks -> absolute nb).
  * ACHILLES -> `oracle_signal(oracle_npz, sd)`: a probe-agnostic fs_rich oracle npz
                (lep / prot_p4 / pi_p4 / pi_pid / n_other_meson, weight w * weight_to_nb -> absolute nb).

The SignalDef selects the topology (pion_id "none" = CC0pi, "pip" = CC1pi) and the acceptance windows
(mu_win + cos_mu|cth, p_win + cth, pi_win + cth).  The SAME cuts and the SAME validated STV formulas
(bank_plot.pN_1pi/dptt_1pi/dpt_1pi/dat_1pi -- with the pion 4-vector set to zero, the CC1pi formulas
reduce EXACTLY to CC0pi) run on both sides.  This replaces the retired engine-bank
select_signal/select_reference (P-A); it generalizes the T2K/MINERvA/MicroBooNE scratch drivers.
"""
import numpy as np

from adonis.reweight import bank_plot as BP
from adonis.constants import PDG_MESONS

_MESONS = list(PDG_MESONS)   # np.isin needs a list/tuple -- a frozenset silently matches NOTHING


def _cos(p4, mom):
    return np.where(mom > 0, p4[:, 3] / np.maximum(mom, 1e-9), -2.0)


def _obs(mu, lead, pip):
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
            "lp_p": pl, "cos_lp": clp, "th_lp": np.degrees(np.arccos(np.clip(clp, -1.0, 1.0))),
            "ppi": ppi, "cos_pi": cpi}


def _mu_pass(pmu, cmu, sd):
    cut = sd.cos_mu if sd.cos_mu is not None else sd.cth       # CC0pi backward/forward mu cut, else cth
    return (pmu >= sd.mu_win[0]) & (pmu <= sd.mu_win[1]) & (cmu > cut)


def _finish(obs, sel, w, chan):
    out = {k: v[sel] for k, v in obs.items()}
    out["w"] = np.asarray(w)[sel]
    out["chan"] = np.asarray(chan)[sel]          # 0 = QE, 1 = RES (for the reference QE/RES breakdown)
    return out


# --------------------------------------------------------------------------- ADoNIS paper_banks side
def bank_signal(bank_dir, sd):
    B = BP.load_bank(bank_dir)                                 # w0 already /n_chunks -> absolute nb
    mu = B["k_lep"].astype(np.float64)
    pmu = np.linalg.norm(mu[:, 1:], axis=1); cmu = _cos(mu, pmu)
    if sd.pion_id == "none":                                   # ---- CC0pi ----
        n_meson = BP._event_sum(B, np.isin(B["fs_pid"], _MESONS).astype(float))
        lead, hasp = BP.leading_proton(B)                      # global leading proton (NUISANCE def)
        pip = np.zeros_like(mu)
        pl = np.linalg.norm(lead[:, 1:], axis=1); cl = _cos(lead, pl)
        sel = (hasp & (n_meson == 0) & _mu_pass(pmu, cmu, sd)
               & (pl > sd.p_win[0]) & (pl < sd.p_win[1]) & (cl > sd.cth))
        if sd.proton_count == "eq1":                           # exactly ONE proton in acceptance (CC1p0pi)
            pid = B["fs_pid"]; p4p = B["fs_p4"]; pmp = np.linalg.norm(p4p[:, 1:], axis=1)
            inacc = (pid == 2212) & (pmp > sd.p_win[0]) & (pmp < sd.p_win[1]) & (_cos(p4p, pmp) > sd.cth)
            sel = sel & (BP._event_sum(B, inacc.astype(float)).astype(int) == 1)
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
        sel = (one_pion & hasp & _mu_pass(pmu, cmu, sd)
               & (ppi >= sd.pi_win[0]) & (ppi < sd.pi_win[1]) & (cpi > sd.cth))
    return _finish(_obs(mu, lead, pip), sel, B["w0"], B["channel"])       # channel: 0 QE, 1 RES


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


def bank_signal_nc(bank_dir, sd):
    """NC1pi0(Xp) on an ADoNIS bank: EXACTLY one pi0, no charged pions, no non-pion mesons."""
    B = BP.load_bank(bank_dir)
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
