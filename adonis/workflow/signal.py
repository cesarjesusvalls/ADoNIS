"""Signal selection on the RICH ENGINE bank schema (+ ACHILLES reference), driven by a SignalDef.

ONE `select_signal(bank, sd)` reproduces every inline ADoNIS selection in the scripts:
  - res bank + CC1pi sd  == cc1pi_engine_plot.engine_signal
  - qe  bank + CC1pi sd  == cc1pi_engine_plot.qe_created_signal   (primary pion absent -> created pi+)
  - qe  bank + CC0pi sd  == cc0pi_engine_combined._select_engine  (QE, 0 pion)
  - res bank + CC0pi sd  == cc0pi_engine_combined._select_engine on the RES bank (pion absorbed)
So a channel is just: apply select_signal with the channel's sd to each ADoNIS input bank and merge.

The ADoNIS side targets the rich engine schema (pid_pi, cr_pid, pi_post, cr_p4, prot, struck, mu, nu, w);
it must NOT use cc1pi_signal.ado_select (older schema).  The ACHILLES reference uses its own rich
schema: select_reference reuses cc1pi_signal.ach_select for CC1pi and mirrors the CC0pi rich selection.
"""
from __future__ import annotations
import numpy as np
import adonis.workflow.observables as O

_PIONS = (211, 111, -211)
_OTHER_MESON = (111, -211, -1)            # vs a pi+ signal: pi0 / pi- / converted(eta,K)


def _load(bank):
    return bank if isinstance(bank, dict) else dict(np.load(bank))


def _acc(p4, win, cth):
    m = np.linalg.norm(p4[:, 1:], axis=1)
    return (m > win[0]) & (m < win[1]) & (p4[:, 3] / np.clip(m, 1e-9, None) > cth)


def _muon_mask(mu, sd):
    pmu = np.linalg.norm(mu[:, 1:], axis=1); cmu = mu[:, 3] / np.clip(pmu, 1e-9, None)
    if sd.cos_mu is not None:              # CC0pi: momentum lower-bound + backward cos cut
        return (pmu > sd.mu_win[0]) & (pmu < sd.mu_win[1]) & (cmu > sd.cos_mu)
    return _acc(mu, sd.mu_win, sd.cth)     # CC1pi: full window + forward cos


def _lead_proton(prot, sd):
    """Returns (lead_p4 (n,4), has_p (n,)).  in_window: leading among in-window top-M;
    global: global max-|p| proton that must itself pass the window (T2K NUISANCE def)."""
    n = prot.shape[0]; ar = np.arange(n)
    pm = np.linalg.norm(prot[:, :, 1:], axis=2); cth = prot[:, :, 3] / np.clip(pm, 1e-9, None)
    inwin = (pm > sd.p_win[0]) & (pm < sd.p_win[1]) & (cth > sd.cth)
    if sd.proton_lead == "global":
        j = np.argmax(pm, axis=1); lead = prot[ar, j]
        return lead, inwin[ar, j]
    mm = np.where(inwin, pm, -1.0); j = np.argmax(mm, axis=1)
    return prot[ar, j], mm[ar, j] > 0


def _pion(b, sd):
    """Returns (meson_ok (n,), pi_f (n,4) or None).  pip: exactly one pi+ / no other meson over
    {primary, created}; anypi: >=1 surviving pion; none: NO surviving pion (CC0pi veto)."""
    pp, cp = b["pid_pi"], b["cr_pid"]
    if sd.pion_id == "none":
        return (~np.isin(pp, _PIONS)) & (~np.isin(cp, _PIONS)), None
    if sd.pion_id == "anypi":
        prim = np.isin(pp, _PIONS)
        pi_f = np.where(prim[:, None], b["pi_post"], b["cr_p4"])
        return (prim | np.isin(cp, _PIONS)), pi_f
    prim_pip = (pp == 211); cr_pip = (cp == 211)              # "pip"
    n_pip = prim_pip.astype(int) + cr_pip.astype(int)
    n_other = np.isin(pp, _OTHER_MESON).astype(int) + np.isin(cp, _OTHER_MESON).astype(int)
    pi_f = np.where(prim_pip[:, None], b["pi_post"], b["cr_p4"])
    return (n_pip == 1) & (n_other == 0), pi_f


def _n_ejected(prot, thr):
    """Count final-state protons with |p| > thr (the ejected-proton multiplicity)."""
    return (np.linalg.norm(prot[:, :, 1:], axis=2) > thr).sum(1)


def select_signal(bank, sd, carbon_only=True):
    """Apply signal topology `sd` to a rich engine bank -> observable dict (keys depend on channel).
    sd.n_ejected (if set) requires EXACTLY that many ejected protons; n_ejected=0 (no proton) yields
    muon-only observables (Q2, p_mu, cos_mu)."""
    b = _load(bank); n = len(b["w"]); mu, nu, struck = b["mu"], b["nu"], b["struck"]
    meson_ok, pi_f = _pion(b, sd)
    best, has_p = _lead_proton(b["prot"], sd)
    sel = meson_ok & (b["w"] > 0) & _muon_mask(mu, sd)
    if sd.n_ejected is not None:
        sel = sel & (_n_ejected(b["prot"], sd.eject_thresh) == sd.n_ejected)
    if sd.require_proton:
        sel = sel & has_p
    if pi_f is not None:
        sel = sel & _acc(pi_f, sd.pi_win, sd.cth)
    if carbon_only:
        sel = sel & (np.linalg.norm(struck[:, 1:], axis=1) > 1.0)
    s = sel
    proton_obs = sd.require_proton or (sd.n_ejected is not None and sd.n_ejected >= 1)
    if pi_f is not None:                                      # CC1pi TKI observables
        dptt, pn, dat, _ = O.tki(mu[s], pi_f[s], best[s], np.zeros(int(s.sum()), bool), 0)
        W, Q2 = O.vertex_W_Q2(nu[s], mu[s], struck[s])
        out = dict(pn=pn, dptt=dptt, dalphat=dat, W=W, Q2=Q2,
                   pi_p=O.mom(pi_f[s]), lp_p=O.mom(best[s]))
    elif proton_obs:                                          # CC0pi observables (leading proton)
        out = O.cc0pi_obs(mu[s], best[s], struck[s], nu[s])
    else:                                                     # 0-proton topology: muon-only
        out = O.muon_obs(mu[s], nu[s])
    out["w"] = b["w"][s]
    return out


# ----------------------------------------------------------------------------- ACHILLES reference
def _sd_to_legacy(sd):
    """Map a SignalDef -> the cc1pi_signal.DEFAULT-style dict that ach_select consumes (CC1pi)."""
    return dict(mu_win=sd.mu_win, pi_win=sd.pi_win, p_win=sd.p_win, cth=sd.cth, fsi=True,
                proton_source="native+knockout", count_recoil_neutron=sd.count_recoil_neutron,
                require_proton=sd.require_proton, proton_count=sd.proton_count, pion_id=sd.pion_id,
                target=sd.target, W_conv=sd.W_conv)


def _ach_cc0pi(b, sd, weight_to_nb, carbon_only=True):
    """CC0pi selection on the ACHILLES rich bank (mirrors cc0pi_engine_combined._select_ach)."""
    n = len(b["w"]); ar = np.arange(n)
    mu, nu, struck = b["mu"], b["nu"], b["struck"]
    pmu = np.linalg.norm(mu[:, 1:], axis=1); cmu = mu[:, 3] / np.clip(pmu, 1e-9, None)
    npi = np.isin(b["pi_pid"], list(_PIONS)).sum(1)
    pr = b["prot_p4"]; pm = np.linalg.norm(pr[:, :, 1:], axis=2)
    j = np.argmax(pm, axis=1); lead = pr[ar, j]; lpm = pm[ar, j]
    lcth = lead[:, 3] / np.clip(lpm, 1e-9, None)
    in_win = (lpm > sd.p_win[0]) & (lpm < sd.p_win[1]) & (lcth > sd.cth)
    sel = (b["w"] > 0) & (npi == 0) & (pmu > sd.mu_win[0]) & (cmu > sd.cos_mu)
    if sd.ref_proc is not None:
        if "proc" not in b:
            raise ValueError("ref_proc set but reference bank has no 'proc' field; "
                             "re-extract with scripts/extract_cc1pi_rich.py")
        sel = sel & np.isin(b["proc"], list(sd.ref_proc))
    if sd.n_ejected is not None:
        sel = sel & (_n_ejected(pr, sd.eject_thresh) == sd.n_ejected)
    if sd.require_proton:
        sel = sel & in_win
    if carbon_only:
        sel = sel & (np.linalg.norm(struck[:, 1:], axis=1) > 1.0)
    s = sel
    if sd.require_proton or (sd.n_ejected is not None and sd.n_ejected >= 1):
        out = O.cc0pi_obs(mu[s], lead[s], struck[s], nu[s])
    else:                                                     # 0-proton topology: muon-only
        out = O.muon_obs(mu[s], nu[s])
    out["w"] = b["w"][s] * weight_to_nb
    return out


def select_reference(bank, sd, carbon_only=True):
    """ACHILLES reference selection -> observable dict on the absolute nb scale.  CC1pi reuses
    cc1pi_signal.ach_select; CC0pi (pion_id='none') uses the rich-bank CC0pi selection."""
    b = _load(bank); wnb = float(b["weight_to_nb"])
    if sd.pion_id == "none":
        return _ach_cc0pi(b, sd, wnb, carbon_only)
    if sd.ref_proc is not None:                                # restrict to QE/RES process ids
        if "proc" not in b:
            raise ValueError("ref_proc set but reference bank has no 'proc' field; "
                             "re-extract with scripts/extract_cc1pi_rich.py")
        n = len(b["w"]); m = np.isin(b["proc"], list(sd.ref_proc))
        b = {k: (v[m] if hasattr(v, "shape") and v.shape[:1] == (n,) else v) for k, v in b.items()}
    import scripts.cc1pi_signal as S
    H = S.ach_select(b, _sd_to_legacy(sd)); H["w"] = H["w"] * wnb
    return H
