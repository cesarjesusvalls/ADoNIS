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
import numpy as np

PION_PIDS = (211, 111, -211)


_FS = ("fs_off", "fs_pid", "fs_chg", "fs_p4")


def load_bank(outdir, max_chunks=None):
    """Concatenate all chunks into one in-memory FULL-RECORD bank (kinematics + ragged final state + bare w0
    + hard-vertex amps2 records + FSI kind-1 record).  Only w0 is divided by n_chunks (-> sum = cross
    section); the records are per-event multipliers.  Exact reweight at any theta via bank_reweight.

    max_chunks: load only the first N chunk files (a subsampled bank).  w0 is then divided by the number
    LOADED (not the manifest total), so the central stays a proper cross-section estimate and the MC-error
    FRACTION reflects the loaded statistics -- exactly what a closure/fit needs when the full bank is far
    larger than the MC precision required."""
    man = json.load(open(f"{outdir}/manifest.json"))
    nchunks = man["n_chunks"]; files = sorted(glob.glob(f"{outdir}/chunk_*.npz"))
    if max_chunks is not None:
        files = files[:max_chunks]; nchunks = len(files)   # divide by loaded count (subsampled bank)
    perev = {}; fs_pid = []; fs_chg = []; fs_p4 = []; offs = [np.array([0], np.int64)]
    ev_off = 0                       # events seen so far -> shifts each chunk's ragged indices
    for f in files:
        d = np.load(f)
        # --- TEMPORARY k_lep compat shim (REMOVE after banks are regenerated/migrated) --------------
        # sec1 renamed the outgoing-lepton field k_mu (CC) / k_e (EM) -> k_lep.  Banks written before
        # that rename still carry the old name on disk.  Alias it to k_lep PER CHUNK, before
        # concatenation, so a directory caught mid-migration (some chunks already k_lep, some not --
        # migrate_k_lep is atomic per chunk) still assembles a full-length k_lep aligned with w0.
        # A global post-concat alias would populate k_lep from only the migrated chunks -> length
        # mismatch against w0.  Delete this once every bank under $ADONIS_OUT carries k_lep.
        _lep_alias = None
        if "k_lep" not in d.files:
            _lep_alias = next((o for o in ("k_mu", "k_e") if o in d.files), None)
        # -------------------------------------------------------------------------------------------
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
            perev.setdefault("k_lep" if key == _lep_alias else key, []).append(v)
        fs_pid.append(d["fs_pid"]); fs_chg.append(d["fs_chg"]); fs_p4.append(d["fs_p4"])
        offs.append(offs[-1][-1] + d["fs_off"][1:])
        ev_off += len(d["w0"])
    B = {k: np.concatenate(v) for k, v in perev.items()}
    B["w0"] = B["w0"] / nchunks
    B["fs_pid"] = np.concatenate(fs_pid); B["fs_chg"] = np.concatenate(fs_chg)
    B["fs_p4"] = np.concatenate(fs_p4); B["fs_off"] = np.concatenate(offs)
    B["n_chunks"] = nchunks
    _coerce_fsi_dtypes(B)                               # empty (free-H) banks default bool/int FSI to float32
    from adonis.reweight.reweight_model import nominal_knobs, knob_specs
    B["labels"] = [s[2] for s in knob_specs(nominal_knobs())]   # plotted knobs (no pw_norm, no sscat)
    n = len(B["w0"]); B["_eidx"] = np.repeat(np.arange(n), np.diff(B["fs_off"]))
    return B


def bank_nchunks(outdir):
    """The manifest n_chunks (the w0 divisor -> absolute nb), without loading any events."""
    return json.load(open(f"{outdir}/manifest.json"))["n_chunks"]


# Canonical dtypes for the integer/bool FSI records.  An EMPTY bank (e.g. free hydrogen: a free proton has
# no nucleon cascade, so every f_* nucleon/pion record is length 0) lets numpy default those arrays to
# float32 -- and jax then rejects a float-typed `iso` used as an index (se[iso]).  Coercing to the canonical
# dtypes at load is a no-op on a populated carbon bank and fixes the empty-bank schema.
_FSI_CANON = (("f_iso", np.int8), ("f_bc", np.int8), ("f_hh", bool), ("f_inel", bool),
              ("f_swap", bool), ("f_pi_hh", bool), ("f_p_eidx", np.int32), ("f_n_eidx", np.int32))


def _coerce_fsi_dtypes(B):
    for _k, _dt in _FSI_CANON:
        if _k in B and B[_k].dtype != np.dtype(_dt):
            B[_k] = B[_k].astype(_dt)
    return B


# The ragged FSI slot records (bank_reweight._FSI_F) split into a PION family (indexed by f_p_eidx) and a
# NUCLEON family (indexed by f_n_eidx); filter_events gathers each by its own event index.
_FSI_PION = ("bc", "sa", "ss_el", "ss", "si", "pi_hh", "pi_a", "sa_c", "ss_el_c", "ss_c", "si_c")
_FSI_NUC = ("hh", "a", "iso", "finel", "inel", "swap")


def filter_events(B, keep):
    """Compact full-record bank of ONLY the events where keep[i] is True.

    Per-event fields (w0, k_lep, the hard-vertex hv_qe_/hv_res_ records, ...) are gathered by event; the
    ragged families -- final state (fs_off/fs_pid/fs_chg/fs_p4), the ks_ summary (ks_eidx), and the two FSI
    slot families (pion via f_p_eidx, nucleon via f_n_eidx) -- keep only the selected events' slots with
    their event index REMAPPED to the new 0..M-1 order, and fs_off/_eidx are rebuilt.  So the compact bank is
    self-consistent for every downstream reducer AND bank_weight(filter_events(B, sig), theta) equals
    bank_weight(B, theta)[sig] -- i.e. a fit can cache N_selected instead of N_total (see workflow.selection
    .select_bank for the streaming, full-bank version)."""
    keep = np.asarray(keep, bool); idx = np.where(keep)[0]; n = len(B["w0"])
    remap = np.full(n, -1, np.int64); remap[idx] = np.arange(len(idx))
    out = {}
    # ragged final state: rebuild fs_off, gather the selected events' particles
    off = np.asarray(B["fs_off"]); lens = np.diff(off)[idx]
    new_off = np.concatenate([[0], np.cumsum(lens)]).astype(off.dtype)
    part = (np.concatenate([np.arange(off[e], off[e + 1]) for e in idx]) if len(idx)
            else np.zeros(0, np.int64))
    out["fs_off"] = new_off
    for k in ("fs_pid", "fs_chg", "fs_p4"):
        out[k] = np.asarray(B[k])[part]
    # eidx-indexed ragged families: ks_ summary + the two FSI slot families
    fams = []
    if "ks_eidx" in B:
        fams.append(("ks_eidx", [k for k in B if k.startswith("ks_") and k != "ks_eidx"]))
    fams.append(("f_p_eidx", [f"f_{x}" for x in _FSI_PION]))
    fams.append(("f_n_eidx", [f"f_{x}" for x in _FSI_NUC]))
    for eidxname, fields in fams:
        if eidxname not in B:
            continue
        e = np.asarray(B[eidxname]); m = remap[e] >= 0
        out[eidxname] = remap[e[m]]
        for x in fields:
            out[x] = np.asarray(B[x])[m]
    # per-event fields (length n): gather by event; scalars / grids pass through
    handled = set(out) | {"_eidx"}
    for k, v in B.items():
        if k in handled:
            continue
        a = np.asarray(v)
        out[k] = a[idx] if (a.ndim >= 1 and a.shape[0] == n) else v
    out["_eidx"] = np.repeat(np.arange(len(idx)), np.diff(new_off))
    return out


def load_bank_chunk(f, nchunks):
    """ONE chunk file as a full-record bank dict -- same schema as load_bank for a single chunk (ev_off=0,
    so its ragged indices are already 0-based / self-contained).  w0 is divided by `nchunks` (the manifest
    total, passed in) so that SUMMING a per-chunk reduction over all chunks reproduces load_bank's
    cross-section exactly.  This is what the streaming, bounded-memory selection in workflow.selection uses:
    it loads one chunk, keeps only the ~1% of events that pass the cut, frees the chunk, and never holds the
    whole concatenated bank in memory (which OOMs on the 20M-event Ar/uBooNE banks)."""
    d = np.load(f)
    _lep_alias = None if "k_lep" in d.files else next((o for o in ("k_mu", "k_e") if o in d.files), None)
    B = {}
    for key in d.files:                                # per-event records (skip the ragged final-state keys)
        if key in _FS:
            continue
        B["k_lep" if key == _lep_alias else key] = d[key]   # f_p_eidx/f_n_eidx need no ev_off shift (ev_off=0)
    B["w0"] = B["w0"] / nchunks
    B["fs_pid"] = d["fs_pid"]; B["fs_chg"] = d["fs_chg"]; B["fs_p4"] = d["fs_p4"]; B["fs_off"] = d["fs_off"]
    B["n_chunks"] = nchunks
    _coerce_fsi_dtypes(B)                               # empty (free-H) banks default bool/int FSI to float32
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


def pion_counts(B):
    """(n_pi+, n_pi0, n_pi-) per event from the final state."""
    pid = B["fs_pid"]; eidx = B["_eidx"]; n = len(B["w0"])
    return tuple(np.bincount(eidx, weights=(pid == p).astype(float), minlength=n).astype(int)
                 for p in (211, 111, -211))


def single_pip(B):
    """p4 of the (single) surviving pi+ per event (n,4); for >1 pi+ the last wins (CC1pi requires ==1)."""
    return _single_pion(B, 211)


def single_pi0(B):
    """p4 of the (single) surviving pi0 per event (n,4) -- the NC1pi0 signal pion.

    Same contract as single_pip, and for the same reason: the selection requires exactly one, so the
    ">1 -> last wins" case is unreachable on signal.  Kept as a thin twin rather than a pid argument
    on a shared helper so both call sites read as the physics they mean."""
    return _single_pion(B, 111)


def _single_pion(B, pid_want):
    pid = B["fs_pid"]; p4 = B["fs_p4"]; eidx = B["_eidx"]; n = len(B["w0"])
    sel = pid == pid_want; out = np.zeros((n, 4)); out[eidx[sel]] = p4[sel]
    return out



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


def dphit_1pi(kmu, lead, pip):
    """Transverse DEFLECTING angle delta_phiT [rad] = angle between the transverse muon and the
    transverse HADRON system: arccos(-p_T^mu . p_T^had / (|p_T^mu||p_T^had|)).

    NB this is NOT delta_alphaT: dat measures the muon against the transverse IMBALANCE
    (p_T^mu + p_T^had), dphit measures it against the hadron system itself (pip is zero for CC0pi,
    so the hadron system is just the leading proton).  Same convention as kinematics.delta_phiT,
    which is the EventRecord-level twin of this bank-level primitive."""
    lt = kmu[:, 1:3]; ht = lead[:, 1:3] + pip[:, 1:3]
    num = -np.sum(lt * ht, axis=1)
    den = np.linalg.norm(lt, axis=1) * np.clip(np.linalg.norm(ht, axis=1), 1e-9, None)
    return np.arccos(np.clip(num / den, -1.0, 1.0))


# ---- histograms (accumulate in f64) ----------------------------------------------------------------- #
def hist_forward(values, B, mask, edges, conv=1.0):
    h, _ = np.histogram(values[mask], bins=edges, weights=B["w0"][mask].astype(np.float64))
    return h / np.diff(edges) * conv


if __name__ == "__main__":
    B = load_bank(sys.argv[1] if len(sys.argv) > 1 else "output/event_bank")
    print(f"loaded {len(B['w0'])} events ({(B['channel']==0).sum()} QE + {(B['channel']==1).sum()} RES), "
          f"{len(B['fs_pid'])} final-state particles, {B['n_chunks']} chunk(s)", flush=True)
