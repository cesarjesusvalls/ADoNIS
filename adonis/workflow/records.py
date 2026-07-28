"""THE centralized cascade-outcome record builder -- one place, every probe.

`cascade_outcome_record(out, prim_fate, fsi_rec, primary)` takes the RAW cascade output batch (the same
`out` batch run_cascade_pool returns and cascade_nucleus exposes as nterms[0]) and produces the UNIFORM
record set that every bank carries, regardless of whether the primary was a neutrino vertex, an electron
vertex, or a tagged hadron beam:

  * fs_*   : full ragged final state (fs_off, fs_pid, fs_chg, fs_p4) -- eta-aware (pion charge 3 -> 221)
  * ks_*   : the escaped-particle list as species/charge/|p|/cos_theta (angular view of fs_*)
  * n_*    : final-state multiplicities (n_p/n_n out, n_pi+/pi0/pi-/eta)
  * reacted/absorbed : primary-interaction flags derived from the fates (meaningful for ANY primary)
  * f_*    : the FSI kind-1 reweight record (via compact_fsi_record) -- accepts one record or a list to
             concatenate (weak/EM bank the qe+res blocks; hadron banks a single block)

This replaces the two duplicated copies of this logic (analysis/paper/beams/beam_bank.py::build and
adonis/workflow/reweight_bank.py).  No scatter: the helpers live here.
"""
from __future__ import annotations
import numpy as np

import adonis.fsi.cascade as CF
from adonis.fsi.cascade import (PION, NUCLEON, FATE_NONE, FATE_ABSORB, FATE_CONVERT, FATE_CAPTURE,
                                _TRACK_OFFSET)

PMAX = 1e4   # MeV ceiling: drop the rare (~0.1%) cascade-artifact nucleons (inf/sentinel momenta)
_PI_PID = np.array([211, 111, -211, 221])    # pion charge idx 0:+ 1:0 2:- 3:eta (the eta rides the PION slot)


# --------------------------------------------------------------------------- final state (eta-aware) ---
def final_state(out):
    """Ragged (fs_off, fs_pid, fs_chg, fs_p4) over the ALIVE escaped particles.  Eta-aware (charge 3 ->
    pid 221).  ONE definition for every probe (was duplicated, and the neutrino/(e,e') copy mislabelled a
    surviving eta as a pi- via clip(charge,0,2))."""
    sp = np.asarray(out["species"]); chg = np.asarray(out["charge"]); al = np.asarray(out["alive"])
    p4 = np.asarray(out["p4"]).astype(np.float64)
    pid = np.where(sp == PION, _PI_PID[np.clip(chg, 0, 3)], np.where(chg == 1, 2212, 2112))
    mom = np.sqrt(np.nan_to_num(p4[:, :, 1:] ** 2, posinf=np.inf).sum(2))
    good = al & np.isfinite(p4).all(2) & (mom < PMAX)
    cnt = good.sum(1).astype(np.int64); off = np.concatenate([[0], np.cumsum(cnt)]); m = good.reshape(-1)
    return dict(fs_off=off.astype(np.int64), fs_pid=pid.reshape(-1)[m].astype(np.int32),
                fs_chg=chg.reshape(-1)[m].astype(np.int32), fs_p4=p4.reshape(-1, 4)[m].astype(np.float32),
                _ndrop=int((al & ~good).sum()))


def merge_fs_off(offs):
    """Concatenate several fs_off blocks (qe then res) into one offset array."""
    off = offs[0]
    for b in offs[1:]:
        off = np.concatenate([off, off[-1] + b[1:]])
    return off.astype(np.int64)


# --------------------------------------------------------------------------- multiplicities + ks_* -----
def multiplicities(out):
    """Charge-resolved escaped multiplicities (n_p_out/n_n_out + n_pi+/pi0/pi-/eta)."""
    sp = np.asarray(out["species"]); chg = np.asarray(out["charge"]); al = np.asarray(out["alive"])
    isN = (sp == NUCLEON) & al; isPi = (sp == PION) & al
    i16 = lambda a: a.sum(1).astype(np.int16)
    return dict(n_p_out=i16(isN & (chg == 1)), n_n_out=i16(isN & (chg == 0)),
                n_pip=i16(isPi & (chg == 0)), n_pi0=i16(isPi & (chg == 1)),
                n_pim=i16(isPi & (chg == 2)), n_eta=i16(isPi & (chg == 3)),
                n_pi_out=i16(isPi & (chg != 3)))


def kicked_secondaries(out):
    """Flat per-escaped-particle kinematics: species/charge/|p|/cos_theta(+z) + primary tag + event idx.
    ks_prim tags the escaped PRIMARIES via track_id (< _TRACK_OFFSET) -- the ONE uniform primary criterion
    for every probe.  (gen==0 over-counted knocked-out background nucleons; origin==_ORIG_PRIM_PI was
    hadron-beam specific -- track_id subsumes both and equals the beam reference byte-for-byte.)"""
    sp = np.asarray(out["species"]); chg = np.asarray(out["charge"]); al = np.asarray(out["alive"])
    tid = np.asarray(out["track_id"]); p3 = np.asarray(out["p4"])[:, :, 1:]
    pm = np.linalg.norm(p3, axis=2); cth = p3[:, :, 2] / np.clip(pm, 1e-9, None)
    eidx = np.broadcast_to(np.arange(sp.shape[0], dtype=np.int32)[:, None], al.shape)
    return dict(ks_eidx=eidx[al].astype(np.int32), ks_species=sp[al].astype(np.int8),
                ks_charge=chg[al].astype(np.int8), ks_pmag=pm[al].astype(np.float32),
                ks_cth=cth[al].astype(np.float32), ks_prim=(tid[al] < _TRACK_OFFSET))


# --------------------------------------------------------------------------- primary scatter count -----
def primary_nsc(out, prim_fate):
    """The surviving primary's scatter count, per event: nsc of the escaped primary (track_id w), maxed
    over primaries -- 0 for primaries that reacted terminally (not in the final state; their reaction is
    carried by prim_fate).  This + prim_fate ARE the per-primary ground truth; every reaction flag is a
    view over them (see `derive_flags`), so nothing derived is persisted."""
    tid = np.asarray(out["track_id"]); nsc = np.asarray(out["nsc"]); al = np.asarray(out["alive"])
    n, Wp = np.asarray(prim_fate).shape
    nsc_prim = np.zeros(n, np.int16)
    for w in range(Wp):
        nsc_w = (nsc * (al & (tid == w))).max(axis=1)                  # scatters of escaped primary w
        nsc_prim = np.maximum(nsc_prim, nsc_w.astype(np.int16))
    return dict(nsc_prim=nsc_prim)


# --------------------------------------------------------------------------- derived reaction flags ----
def derive_flags(bank):
    """THE single definition of the event-level primary-interaction flags, computed from the ONLY two
    stored ground-truth arrays: per-primary fate `prim_fate` (n, Wmax; FATE_NONE-padded) and per-event
    scatter count `nsc_prim` (n,).  Nothing derived is persisted -- consumers call this.  Fate-derived,
    no species branch (nucleons never carry ABSORB/CONVERT, mesons never CAPTURE, so physics separates
    the flags, not an `if`):
      absorbed = a primary meson was consumed  (fate in {ABSORB, CONVERT} -- true piNN->NN OR pi->eta)
      reacted  = any primary reaction fate  OR  the primary escaped after scattering (nsc_prim>0)
    `bank` is any mapping exposing prim_fate / nsc_prim (e.g. a loaded npz)."""
    pf = np.asarray(bank["prim_fate"]); nsc = np.asarray(bank["nsc_prim"])   # (n, Wmax), (n,)
    absorbed = np.isin(pf, (FATE_ABSORB, FATE_CONVERT)).any(axis=1)
    reacted = np.isin(pf, (FATE_ABSORB, FATE_CONVERT, FATE_CAPTURE)).any(axis=1) | (nsc > 0)
    return dict(reacted=reacted, absorbed=absorbed)


# --------------------------------------------------------------------------- FSI kind-1 record ---------
def fsi_kind1(recs, ns):
    """Concatenate the per-block FLAT FSI kind-1 records (already compacted), offsetting each block's
    per-slot event index by the running event count.  `recs` = list of compact_fsi_record outputs,
    `ns` = the per-block event counts (qe then res for weak/EM; a single block for hadron)."""
    from adonis.fsi.cascade import _P_SLOT, _N_SLOT
    flat = {f: np.concatenate([rc[f] for rc in recs]) for f in _P_SLOT + _N_SLOT}
    for tag in ("p_eidx", "n_eidx"):
        acc, base = [], 0
        for rc, nn in zip(recs, ns):
            acc.append(np.asarray(rc[tag]) + base); base += nn
        flat[tag] = np.concatenate(acc)
    save = {"f_p_eidx": flat["p_eidx"].astype(np.int32), "f_n_eidx": flat["n_eidx"].astype(np.int32)}
    for field, arr in flat.items():
        if field.endswith("_eidx"):
            continue
        dt = (np.int8 if field in ("bc", "iso")
              else (bool if field in ("hh", "inel", "swap", "pi_hh") else np.float32))
        save[f"f_{field}"] = arr.astype(dt)
    return save, len(flat["p_eidx"]), len(flat["n_eidx"])


def cascade_outcome_record(blocks, fsi_recs, ns):
    """The full uniform cascade-outcome record for one chunk, concatenated across the present BLOCKS
    (one block for a hadron beam; the qe then res blocks for a weak/EM bank).  Each block is
    (out, prim_fate): out = raw escaped batch, prim_fate = (nb, Wp) per-primary terminal fate (one
    column per primary track_id).  The ground truth is exactly TWO arrays: `prim_fate` stored as a
    padded rectangle (n, Wmax) -- FATE_NONE fills the shorter (QE, 1-primary) block up to the widest
    (RES, 2-primary) -- and `nsc_prim` (n,).  A padded rectangle is self-describing (no companion count
    array) and concatenates cleanly across chunks (axis 0) under the naive beam loader.  Every reaction
    flag is a view over these two (see `derive_flags`); none is persisted.  Returns (save_dict, meta)."""
    fs_offs, fs_pid, fs_chg, fs_p4, ndrop = [], [], [], [], 0
    acc = {}                                             # multiplicities + ks_* + nsc_prim
    prim_fates = []                                      # per-block (nb, Wp) fate rectangles
    base = 0
    for (out, pfate), nb in zip(blocks, ns):
        fs = final_state(out); ndrop += fs.pop("_ndrop")
        fs_offs.append(fs["fs_off"]); fs_pid.append(fs["fs_pid"]); fs_chg.append(fs["fs_chg"]); fs_p4.append(fs["fs_p4"])
        part = {**multiplicities(out), **kicked_secondaries(out), **primary_nsc(out, pfate)}
        part["ks_eidx"] = part["ks_eidx"] + base         # offset ks event-index into the global numbering
        for k, v in part.items():
            acc.setdefault(k, []).append(v)
        prim_fates.append(np.asarray(pfate).astype(np.int8))
        base += nb
    save = dict(fs_off=merge_fs_off(fs_offs), fs_pid=np.concatenate(fs_pid),
                fs_chg=np.concatenate(fs_chg), fs_p4=np.concatenate(fs_p4))
    save.update({k: np.concatenate(v) for k, v in acc.items()})
    Wmax = max(pf.shape[1] for pf in prim_fates)         # widest channel (RES=2); pad the rest with FATE_NONE
    pad = lambda pf: pf if pf.shape[1] == Wmax else np.pad(
        pf, ((0, 0), (0, Wmax - pf.shape[1])), constant_values=FATE_NONE)
    save["prim_fate"] = np.concatenate([pad(pf) for pf in prim_fates], axis=0)   # (n, Wmax) ground truth
    fsi, npslot, nnslot = fsi_kind1(fsi_recs, ns)
    save.update(fsi)
    return save, dict(ndrop=ndrop, pion_slots=npslot, nucleon_slots=nnslot)
