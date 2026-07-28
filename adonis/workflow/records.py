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
from adonis.fsi.cascade import PION, NUCLEON, FATE_ABSORB, FATE_CONVERT, FATE_CAPTURE, _ORIG_PRIM_PI

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
    """Flat per-escaped-particle kinematics: species/charge/|p|/cos_theta(+z) + primary tag + event idx."""
    sp = np.asarray(out["species"]); chg = np.asarray(out["charge"]); al = np.asarray(out["alive"])
    org = np.asarray(out["origin"]); p3 = np.asarray(out["p4"])[:, :, 1:]
    pm = np.linalg.norm(p3, axis=2); cth = p3[:, :, 2] / np.clip(pm, 1e-9, None)
    eidx = np.broadcast_to(np.arange(sp.shape[0], dtype=np.int32)[:, None], al.shape)
    return dict(ks_eidx=eidx[al].astype(np.int32), ks_species=sp[al].astype(np.int8),
                ks_charge=chg[al].astype(np.int8), ks_pmag=pm[al].astype(np.float32),
                ks_cth=cth[al].astype(np.float32), ks_prim=(org[al] == _ORIG_PRIM_PI))


# --------------------------------------------------------------------------- reacted / absorbed --------
def reaction_flags(out, prim_fate, primary):
    """Primary-interaction flags from the fates -- uniform over probes.  `primary` in {'pion','nucleon'}
    is the tagged/primary species whose reaction we report (a hadron beam's projectile; a ν/e⁻ vertex's
    ejected pion [RES] or proton [QE]).  Reproduces beam_bank's per-species definition byte-for-byte."""
    sp = np.asarray(out["species"]); chg = np.asarray(out["charge"]); al = np.asarray(out["alive"])
    org = np.asarray(out["origin"]); nsc = np.asarray(out["nsc"]); pf = np.asarray(prim_fate)
    is_prim = org == _ORIG_PRIM_PI
    if primary == "pion":
        prim_pi = is_prim & (sp == PION) & (chg != 3) & al
        nsc_prim = (nsc * prim_pi).max(axis=1)
        reacted = (pf == FATE_ABSORB) | (pf == FATE_CONVERT) | (nsc_prim > 0)
        has_pi = ((sp == PION) & (chg != 3) & al).any(axis=1)
        absorbed = reacted & ~has_pi
    else:  # nucleon primary
        prim_N = is_prim & (sp == NUCLEON)
        nsc_prim = (nsc * prim_N).max(axis=1)
        reacted = (nsc_prim > 0) | (pf == FATE_CAPTURE)
        absorbed = np.zeros(sp.shape[0], bool)
    return dict(reacted=reacted, absorbed=absorbed, nsc_prim=nsc_prim.astype(np.int16),
                prim_fate=pf.astype(np.int8))


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


def cascade_outcome_record(out, prim_fate, fsi_recs, ns, primary):
    """The full uniform cascade-outcome record for one chunk (one probe).  `out` = raw escaped batch,
    `fsi_recs`/`ns` = the compacted FSI record(s) + event counts, `primary` = the reacting species tag."""
    save = {}
    save.update(final_state(out)); ndrop = save.pop("_ndrop")
    save.update(multiplicities(out))
    save.update(kicked_secondaries(out))
    save.update(reaction_flags(out, prim_fate, primary))
    fsi, npslot, nnslot = fsi_kind1(fsi_recs, ns)
    save.update(fsi)
    save["_meta"] = dict(ndrop=ndrop, pion_slots=npslot, nucleon_slots=nnslot)
    return save
