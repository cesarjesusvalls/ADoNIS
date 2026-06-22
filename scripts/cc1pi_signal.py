"""Signal-definition re-binning for T2K CC1pi+Np, from RICH per-event banks (no regeneration).

Both banks store full 4-vectors so ANY signal definition is pure re-binning:
  ADoNIS  (scripts/gen_cc1pi_rich.py)  : event-matched PRE- and POST-FSI; every proton candidate
                                         (recoil + knockouts) as a separate 4-vec + species.
  ACHILLES(scripts/extract_cc1pi_rich.py): all final-state pions/protons as 4-vec lists + struck pid.
                                         FSI and no-FSI are separate banks (= post / pre-FSI).

A `sigdef` dict selects the definition; change it -> all diff-xsec + ratios re-render.  The STV
observable formulas come from cc1pi_fig_tki.observables (single source of truth).

  sigdef keys:
    mu_win, pi_win, p_win : (lo,hi) MeV ;  cth : forward cos-theta cut (theta<70deg)
    fsi          : True -> post-FSI (FSI bank) ; False -> pre-FSI (no-FSI bank / ADoNIS primary)
    proton_source: "native+knockout" (ACHILLES-equivalent) | "native" | (ADoNIS only)
    count_recoil_neutron : ADoNIS only -- if True the recoil nucleon counts as a "proton" regardless
                           of species (reproduces the OLD pid_N=2212 bug as a knob)
    target       : "carbon" | "hydrogen" | "CH"
    W_conv       : "vertex" (|q+struck|) -- observable only, not a cut
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scripts.cc1pi_fig_tki as F          # observables() -- single source of truth for STV

COS70 = float(np.cos(np.deg2rad(70.0)))

DEFAULT = dict(mu_win=(250., 7000.), pi_win=(150., 1200.), p_win=(450., 1200.), cth=COS70,
               fsi=True, proton_source="native+knockout", count_recoil_neutron=False,
               require_proton=True, proton_count="ge1", pion_id="pip", target="carbon", W_conv="vertex")
# pion_id: "pip" = require a surviving pi+ ; "anypi" = any surviving pion (pi+/pi0/pi-) -> only
# absorption removes it (charge-exchange does not), so it isolates absorption from charge-exchange.
_PIONS = (211, 111, -211)
_OTHER_MESON = (111, -211, -1)            # vs a pi+ signal: pi0 / pi- / converted(eta,K)


def _mom(p):
    return np.linalg.norm(np.atleast_2d(p)[:, 1:], axis=1)


def _acc(p, win, cth):
    m = _mom(p)
    return (m > win[0]) & (m < win[1]) & (p[:, 3] / np.clip(m, 1e-9, None) > cth)


def _vertexW(nu, mu, struck):
    q = nu - mu; tot = q + struck
    W = np.sqrt(np.clip(tot[:, 0] ** 2 - np.sum(tot[:, 1:] ** 2, axis=1), 0, None))
    Q2 = (np.sum(q[:, 1:] ** 2, axis=1) - q[:, 0] ** 2) / 1e6
    return W, Q2


def _finish(mu, pi, lead, struck, nu, w, mask, sd, is_h, chan):
    """Common observable build for the selected events.  chan = primary channel tag per event
    (2212 = p->p pi+ / struck proton, 2112 = n->n pi+ / struck neutron)."""
    s = mask
    dptt, pn, dat, dpt = F.observables(mu[s], pi[s], lead[s], is_h[s], 0)
    W, Q2 = _vertexW(nu[s], mu[s], struck[s])
    return dict(dptt=dptt, pn=pn, dalphat=dat, dpt=dpt,
                pi_p=_mom(pi[s]), lp_p=_mom(lead[s]), W=W, Q2=Q2, w=w[s], chan=np.asarray(chan)[s])


def ado_select(b, sd):
    """ADoNIS rich bank -> selected signal observables under sigdef sd.

    POST-FSI uses the unified pool proton-candidate set prot[] (every escaped proton terminal across
    cascade generations) -- the validated extraction (cf. adonis.workflow.signal).  PRE-FSI uses the
    single primary recoil nucleon (a proton iff Npid==2212, or always if count_recoil_neutron)."""
    fsi = sd["fsi"]
    pi = b["pi_post"] if fsi else b["pi_pre"]
    pid = b["pid_pi_post"] if fsi else b["ppid_pre"]
    n = len(pid); ar = np.arange(n)
    if fsi:
        prot = b["prot"]; cr_pid = b["cr_pid"]                       # (n,M,4) pool candidates + created pion
    else:
        rec = b["rec_pre"]; rec_isp = (b["Npid"] == 2212)
        rec_is_p = np.ones(n, bool) if sd["count_recoil_neutron"] else rec_isp
        prot = np.where(rec_is_p[:, None], rec, 0.0)[:, None, :]      # single primary candidate
        cr_pid = np.zeros(n, np.int64)                               # no created pion pre-FSI
    pm = np.linalg.norm(prot[:, :, 1:], axis=2); cth = prot[:, :, 3] / np.clip(pm, 1e-9, None)
    inwin = (pm > sd["p_win"][0]) & (pm < sd["p_win"][1]) & (cth > sd["cth"])
    nprot = inwin.sum(1)
    has_p = ({"eq1": nprot == 1, "eq2": nprot == 2}.get(sd.get("proton_count"), nprot >= 1))
    lead = prot[ar, np.argmax(np.where(inwin, pm, -1.0), axis=1)]
    # pion selection over {primary, created}: pip = exactly one pi+ and no other meson; anypi = any pion
    if sd.get("pion_id") == "anypi":
        prim = np.isin(pid, _PIONS); pid_ok = prim | np.isin(cr_pid, _PIONS)
        if fsi:
            pi = np.where(prim[:, None], pi, b["cr_p4"])
    else:
        prim_pip = (pid == 211); cr_pip = (cr_pid == 211)
        n_pip = prim_pip.astype(int) + cr_pip.astype(int)
        n_other = np.isin(pid, _OTHER_MESON).astype(int) + np.isin(cr_pid, _OTHER_MESON).astype(int)
        pid_ok = (n_pip == 1) & (n_other == 0)
        if fsi:
            pi = np.where(prim_pip[:, None], pi, b["cr_p4"])
    pstr_p = _mom(b["struck"])
    tgt = {"carbon": pstr_p > 1.0, "hydrogen": pstr_p <= 1.0, "CH": np.ones(n, bool)}[sd["target"]]
    p_req = has_p if sd.get("require_proton", True) else np.ones(n, bool)
    mask = (pid_ok & p_req & (b["w"] > 0) & tgt
            & _acc(b["mu"], sd["mu_win"], sd["cth"]) & _acc(pi, sd["pi_win"], sd["cth"]))
    is_h = ~(pstr_p > 1.0)
    return _finish(b["mu"], pi, lead, b["struck"], b["nu"], b["w"], mask, sd, is_h, b["ipid"])


def ach_select(b, sd):
    """ACHILLES rich bank -> selected signal observables under sigdef sd.  Pass the FSI bank for
    fsi=True, the no-FSI bank for fsi=False (ACHILLES has no event-matched pre/post in one file)."""
    n = len(b["w"]); K = b["pi_p4"].shape[1]; M = b["prot_p4"].shape[1]
    if sd.get("pion_id") == "anypi":                                 # any surviving pion (absorption-only)
        n_anypi = np.isin(b["pi_pid"], _PIONS).sum(1)
        pi = b["pi_p4"][:, 0]                                        # leading pion (slots sorted by |p|)
        pion_ok = (n_anypi >= 1)
    else:                                                            # exactly one pi+ and no other meson
        pip_is = (b["pi_pid"] == 211)
        n_pip = pip_is.sum(1)
        pi = b["pi_p4"][np.arange(n), np.argmax(pip_is, axis=1)]
        pion_ok = (n_pip == 1) & (b["n_other_meson"] == 0)
    # leading in-window proton
    pacc = np.stack([_acc(b["prot_p4"][:, i], sd["p_win"], sd["cth"]) for i in range(M)], axis=1)
    nprot = pacc.sum(1)
    pc = sd.get("proton_count")
    if pc in ("eq0", "eq1", "eq2"):                                  # EXACTLY N in-window protons
        p_req = (nprot == int(pc[-1]))
    else:
        p_req = (nprot >= 1) if sd.get("require_proton", True) else np.ones(n, bool)
    lead = b["prot_p4"][np.arange(n), np.argmax(_mom(b["prot_p4"].reshape(-1, 4)).reshape(n, M) * pacc, axis=1)]
    pstr_p = _mom(b["struck"])
    tgt = {"carbon": pstr_p > 1.0, "hydrogen": pstr_p <= 1.0, "CH": np.ones(n, bool)}[sd["target"]]
    mask = (pion_ok & p_req & (b["w"] > 0) & tgt
            & _acc(b["mu"], sd["mu_win"], sd["cth"]) & _acc(pi, sd["pi_win"], sd["cth"]))
    is_h = ~(pstr_p > 1.0)
    if pc == "eq0":                                                  # CC1pi 0-proton: pion+muon obs
        s = mask; mu = b["mu"][s]; pmu = _mom(mu); W, Q2 = _vertexW(b["nu"][s], mu, b["struck"][s])
        return dict(W=W, Q2=Q2, pi_p=_mom(pi[s]), p_mu=pmu,
                    cos_mu=mu[:, 3] / np.clip(pmu, 1e-9, None), w=b["w"][s])
    return _finish(b["mu"], pi, lead, b["struck"], b["nu"], b["w"], mask, sd, is_h, b["struck_pid"])
