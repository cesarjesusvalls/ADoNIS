"""Plot-time consumer of the differentiable EVENT BANK (adonis.workflow.generate_bank.generate_bank).  NO JAX cascade here -- everything
is a cheap re-sum over the stored per-event records (kinematics, ragged final state, w0, hard-vertex amps2 +
FSI kind-1 + SF records for the EXACT reweight via bank_reweight).

Pick ANY of these at plot time, no re-running:
  * signal definition  -> a boolean mask over events (topology from the full final state + phase space)
  * observable         -> a per-event value (dpt, dat, p_N, muon kinematics, ...)
  * binning            -> any edges
  * forward histogram  (sum w0)                     : the T2K-style distribution
  * reweight/gradient  (bank_reweight.bank_weight)  : exact w(theta), jax-differentiable in every knob

PIDs: proton 2212, neutron 2112, pions {211,111,-211}.
"""
import os, sys, json, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

PION_PIDS = (211, 111, -211)


_FS = ("fs_off", "fs_pid", "fs_chg", "fs_p4")


def load_bank(outdir):
    """Concatenate all chunks into one in-memory FULL-RECORD bank (kinematics + ragged final state + bare w0
    + hard-vertex amps2 records + FSI kind-1 record).  Only w0 is divided by n_chunks (-> sum = cross
    section); the records are per-event multipliers.  Exact reweight at any theta via bank_reweight."""
    man = json.load(open(f"{outdir}/manifest.json"))
    nchunks = man["n_chunks"]; files = sorted(glob.glob(f"{outdir}/chunk_*.npz"))
    perev = {}; fs_pid = []; fs_chg = []; fs_p4 = []; offs = [np.array([0], np.int64)]
    ev_off = 0                       # events seen so far -> shifts each chunk's ragged indices
    for f in files:
        d = np.load(f)
        for key in d.files:
            if key in _FS:
                continue
            v = d[key]
            if key in ("f_p_eidx", "f_n_eidx"):
                # PER-SLOT event index of the ragged FSI record: each chunk numbers its events from 0, so
                # it must be shifted into the global event numbering before concatenation (exactly what
                # fs_off does for the ragged final state).  Without this, every chunk after the first
                # would silently attribute its FSI slots to the wrong events.
                v = v.astype(np.int64) + ev_off
            perev.setdefault(key, []).append(v)
        fs_pid.append(d["fs_pid"]); fs_chg.append(d["fs_chg"]); fs_p4.append(d["fs_p4"])
        offs.append(offs[-1][-1] + d["fs_off"][1:])
        ev_off += len(d["w0"])
    B = {k: np.concatenate(v) for k, v in perev.items()}
    B["w0"] = B["w0"] / nchunks
    B["fs_pid"] = np.concatenate(fs_pid); B["fs_chg"] = np.concatenate(fs_chg)
    B["fs_p4"] = np.concatenate(fs_p4); B["fs_off"] = np.concatenate(offs)
    B["n_chunks"] = nchunks
    from adonis.reweight.reweight_model import nominal_knobs, knob_specs
    B["labels"] = [s[2] for s in knob_specs(nominal_knobs())]   # plotted knobs (no pw_norm, no sscat)
    n = len(B["w0"]); B["_eidx"] = np.repeat(np.arange(n), np.diff(B["fs_off"]))
    return B


# ---- per-event reductions over the ragged final state ----------------------------------------------- #
def _event_sum(B, per_particle):
    n = len(B["w0"])
    return np.bincount(B["_eidx"], weights=per_particle, minlength=n)


def n_pions(B):
    return _event_sum(B, np.isin(B["fs_pid"], PION_PIDS).astype(float)).astype(int)


def n_protons(B, pmin=0.0, pmax=np.inf):
    mom = np.linalg.norm(B["fs_p4"][:, 1:], axis=1)
    sel = (B["fs_pid"] == 2212) & (mom >= pmin) & (mom < pmax)
    return _event_sum(B, sel.astype(float)).astype(int)


def n_neutrons(B):
    return _event_sum(B, (B["fs_pid"] == 2112).astype(float)).astype(int)


def ranked_mom(B, pid_target, k):
    """|p| of the rank-k (0=leading,1=subleading,...) particle of species pid_target per event.
    Returns (mom (n,), has (n,) = event has a rank-k such particle).  No signal cut."""
    pid = B["fs_pid"]; p4 = B["fs_p4"]; eidx = B["_eidx"]; n = len(B["w0"])
    sel = pid == pid_target
    ev = eidx[sel]; mom = np.linalg.norm(p4[sel, 1:], axis=1)
    out = np.zeros(n); has = np.zeros(n, bool)
    if ev.size:
        order = np.lexsort((-mom, ev)); ev = ev[order]; mom = mom[order]   # within event: |p| descending
        newgrp = np.r_[True, ev[1:] != ev[:-1]]
        grpstart = np.maximum.accumulate(np.where(newgrp, np.arange(ev.size), 0))
        rank = np.arange(ev.size) - grpstart
        msk = rank == k
        out[ev[msk]] = mom[msk]; has[ev[msk]] = True
    return out, has


def leading_proton(B):
    """(lead_p4 (n,4), has_proton (n,)) -- global max-momentum escaped proton per event."""
    n = len(B["w0"]); pid = B["fs_pid"]; p4 = B["fs_p4"]; eidx = B["_eidx"]
    mom = np.linalg.norm(p4[:, 1:], axis=1)
    key = np.where(pid == 2212, mom, -1.0)
    maxk = np.full(n, -1.0); np.maximum.at(maxk, eidx, key)
    lead = np.zeros((n, 4)); islead = (pid == 2212) & (key == maxk[eidx])
    lead[eidx[islead]] = p4[islead]
    return lead, maxk > 0.0


# ---- observables + selections (reuse the validated tune formulas) ------------------------------------ #
def _tune():
    if not hasattr(_tune, "_T"):
        from adonis.reweight import tune as T   # import is now side-effect-free (no argv/data at import)
        _tune._T = T
    return _tune._T


def dpt(B, lead):  return np.asarray(_tune()._dpt(B["k_mu"].astype(float), lead))
def dat(B, lead):  return np.asarray(_tune()._dat(B["k_mu"].astype(float), lead))
def acceptance(B, lead):  return np.asarray(_tune()._sel(B["k_mu"].astype(float), lead))


def signal_cc0pi(B, topological=False):
    """model_hist_full CC0pi (primary pion absorbed: prim_pi_pid==0) + acceptance; topological=True instead
    vetoes ANY surviving pion (true 0-meson final state)."""
    lead, has = leading_proton(B)
    sel = acceptance(B, lead) & has
    pi_ok = (n_pions(B) == 0) if topological else (B["prim_pi_pid"] == 0)
    return sel & pi_ok, lead


def pion_counts(B):
    """(n_pi+, n_pi0, n_pi-) per event from the final state."""
    pid = B["fs_pid"]; eidx = B["_eidx"]; n = len(B["w0"])
    return tuple(np.bincount(eidx, weights=(pid == p).astype(float), minlength=n).astype(int)
                 for p in (211, 111, -211))


def single_pip(B):
    """p4 of the (single) surviving pi+ per event (n,4); for >1 pi+ the last wins (CC1pi requires ==1)."""
    pid = B["fs_pid"]; p4 = B["fs_p4"]; eidx = B["_eidx"]; n = len(B["w0"])
    sel = pid == 211; pip = np.zeros((n, 4)); pip[eidx[sel]] = p4[sel]
    return pip


def signal_cc1pi(B):
    """Topological CC1pi+ : exactly one pi+, no pi0/pi-, and a leading proton.  Returns (mask, lead, pip4)."""
    npip, npi0, npim = pion_counts(B); lead, has = leading_proton(B)
    mask = (npip == 1) & (npi0 == 0) & (npim == 0) & has
    return mask, lead, single_pip(B)


def leading_proton_window(B, pmin, pmax, cth=-1.0):
    """Leading proton with momentum in [pmin,pmax) AND cos(theta)>cth (the T2K CC1pi+Np acceptance picks the
    leading ACCEPTED proton -- window and forward cut both enter the pick, as in workflow.signal._prot_in_window)."""
    pid = B["fs_pid"]; p4 = B["fs_p4"]; eidx = B["_eidx"]; n = len(B["w0"])
    mom = np.linalg.norm(p4[:, 1:], axis=1)
    cz = p4[:, 3] / np.clip(mom, 1e-9, None)
    key = np.where((pid == 2212) & (mom >= pmin) & (mom < pmax) & (cz > cth), mom, -1.0)
    maxk = np.full(n, -1.0); np.maximum.at(maxk, eidx, key)
    lead = np.zeros((n, 4)); islead = (pid == 2212) & (key == maxk[eidx]); lead[eidx[islead]] = p4[islead]
    return lead, maxk > 0.0


# T2K CC1pi+Np STV (PRD 103 112009) -- acceptance windows [MeV] + forward cut + nuclear masses for the p_N
# reconstruction.  All three particles (mu, pi+, leading p) must be forward: cos(theta) > cos(70 deg), as in
# workflow.config.SignalDef(cth=COS70) / make_plots.block_cc1pi_stv.
_MU_LO, _MU_HI = 250.0, 7000.0; _PI_LO, _PI_HI = 150.0, 1200.0; _P_LO, _P_HI = 450.0, 1200.0
from adonis.constants import COS70 as _CTH, M_12C as _M12C, M_11B as _M11B   # single source


def signal_cc1pi_stv(B):
    """T2K CC1pi+Np STV signal: exactly one pi+ (no other meson) + muon/pion/leading-proton in acceptance
    (momentum windows + cos(theta)>cos70 on all three).  Returns (mask, lead, pip4)."""
    npip, npi0, npim = pion_counts(B); pip = single_pip(B)
    lead, hasp = leading_proton_window(B, _P_LO, _P_HI, cth=_CTH)
    kmu = B["k_mu"].astype(np.float64)
    pmu = np.linalg.norm(kmu[:, 1:], axis=1); cmu = kmu[:, 3] / np.clip(pmu, 1e-9, None)
    ppi = np.linalg.norm(pip[:, 1:], axis=1); cpi = pip[:, 3] / np.clip(ppi, 1e-9, None)
    mask = ((npip == 1) & (npi0 == 0) & (npim == 0) & hasp
            & (pmu >= _MU_LO) & (pmu < _MU_HI) & (cmu > _CTH)
            & (ppi >= _PI_LO) & (ppi < _PI_HI) & (cpi > _CTH))
    return mask, lead, pip


def dptt_1pi(kmu, lead, pip):
    """Double-transverse momentum imbalance delta_pTT = (p_pi+p_p).zhat, zhat=(beam x p_mu)/|.| (MeV/c)."""
    beam = np.array([0.0, 0.0, 1.0]); mu3 = kmu[:, 1:]; had3 = pip[:, 1:] + lead[:, 1:]
    zhat = np.cross(np.broadcast_to(beam, mu3.shape), mu3)
    zhat = zhat / (np.linalg.norm(zhat, axis=1, keepdims=True) + 1e-9)
    return np.sum(had3 * zhat, axis=1)


def pN_1pi(kmu, lead, pip):
    """Inferred struck-nucleon momentum p_N (carbon-mass reconstruction, NUISANCE/T2K prescription) [MeV]."""
    dptmag = dpt_1pi(kmu, lead, pip)
    pL = kmu[:, 3] + pip[:, 3] + lead[:, 3]; Evis = kmu[:, 0] + pip[:, 0] + lead[:, 0]
    R = _M12C + pL - Evis
    dpL = 0.5 * R - (_M11B ** 2 + dptmag ** 2) / (2.0 * np.maximum(R, 1.0))
    return np.sqrt(np.maximum(dptmag ** 2 + dpL ** 2, 0.0))


def dpt_1pi(kmu, lead, pip):
    """CC1pi+ transverse-momentum imbalance |p_T^mu + p_T^p + p_T^pi| (MeV)."""
    dv = kmu[:, 1:3] + lead[:, 1:3] + pip[:, 1:3]
    return np.linalg.norm(dv, axis=1)


def dat_1pi(kmu, lead, pip):
    """CC1pi+ delta_alphaT [rad] (muon vs the full muon+proton+pion imbalance)."""
    lt = kmu[:, 1:3]; dv = lt + lead[:, 1:3] + pip[:, 1:3]
    num = -np.sum(lt * dv, axis=1)
    den = np.linalg.norm(lt, axis=1) * np.clip(np.linalg.norm(dv, axis=1), 1e-9, None)
    return np.arccos(np.clip(num / den, -1.0, 1.0))


# ---- histograms (accumulate in f64) ----------------------------------------------------------------- #
def hist_forward(values, B, mask, edges, conv=1.0):
    h, _ = np.histogram(values[mask], bins=edges, weights=B["w0"][mask].astype(np.float64))
    return h / np.diff(edges) * conv


if __name__ == "__main__":
    B = load_bank(sys.argv[1] if len(sys.argv) > 1 else "output/event_bank")
    print(f"loaded {len(B['w0'])} events ({(B['channel']==0).sum()} QE + {(B['channel']==1).sum()} RES), "
          f"{len(B['fs_pid'])} final-state particles, {B['n_chunks']} chunk(s)", flush=True)
