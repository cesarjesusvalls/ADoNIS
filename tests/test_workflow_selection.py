"""Fast synthetic-fixture tests for the workflow selection adapter (adonis.workflow.selection):
bank_signal (ADoNIS paper_banks schema) and oracle_signal (ACHILLES fs_rich schema), exercising the CC0pi
and CC1pi+ topologies + the proton_count="eq1" (CC1p0pi) path.  The bank and oracle are hand-built in
tmp_path -- no data files, no cascade -- so this pins the selection contract cheaply.
"""
import json

import numpy as np

from adonis.workflow.config import SignalDef
from adonis.workflow import selection as SG

MU, MP, MPI, MPI0 = 105.658, 938.272, 139.570, 134.977
_CHG = {2212: 1, 2112: 0, 211: 1, -211: -1, 111: 0}
_DIR = np.array([0.05, 0.03, 0.9984]); _DIR = _DIR / np.linalg.norm(_DIR)   # near-forward (cos ~ 0.998)


def _p4(p, m):
    v = p * _DIR
    return np.array([np.sqrt(m ** 2 + p ** 2), v[0], v[1], v[2]])


# 4 events: [0] CC0pi-1p, [1] CC0pi w/ a pi0 (meson veto), [2] CC0pi-2p, [3] CC1pi+ (1 pi+ + 1 p)
_EVENTS = [
    [(2212, 700, MP)],
    [(2212, 700, MP), (111, 200, MPI0)],
    [(2212, 700, MP), (2212, 600, MP)],
    [(2212, 700, MP), (211, 400, MPI)],
]
_CC0PI = SignalDef(mu_win=(250., 7000.), cos_mu=-0.6, p_win=(450., 1000.), cth=0.4,
                   pi_win=None, pion_id="none", proton_lead="global")
_CC0PI_EQ1 = SignalDef(mu_win=(250., 7000.), cos_mu=-0.6, p_win=(450., 1000.), cth=0.4,
                       pi_win=None, pion_id="none", proton_lead="global", proton_count="eq1")
_CC1PI = SignalDef(mu_win=(250., 7000.), p_win=(450., 1200.), pi_win=(150., 1200.), cth=0.342,
                   cos_mu=None, pion_id="pip", proton_lead="in_window")
_KEYS = {"dpt", "dalphat", "pn", "dptt", "w", "chan"}


def _write_bank(tmp):
    fs_pid, fs_chg, fs_p4, off = [], [], [], [0]
    for ev in _EVENTS:
        for pid, p, m in ev:
            fs_pid.append(pid); fs_chg.append(_CHG[pid]); fs_p4.append(_p4(p, m))
        off.append(off[-1] + len(ev))
    np.savez(tmp / "chunk_0.npz",
             k_mu=np.array([_p4(500, MU)] * 4, float), w0=np.ones(4), channel=np.array([0, 1, 0, 1]),
             fs_off=np.array(off, np.int64), fs_pid=np.array(fs_pid, int),
             fs_chg=np.array(fs_chg, int), fs_p4=np.array(fs_p4, float))
    json.dump({"n_chunks": 1}, open(tmp / "manifest.json", "w"))
    return str(tmp)


def _write_oracle(tmp):
    M = 2
    prot = np.zeros((4, M, 4)); pip = np.zeros((4, M, 4)); pid = np.zeros((4, M), int)
    prot[0, 0] = _p4(700, MP)
    prot[1, 0] = _p4(700, MP); pid[1, 0] = 111; pip[1, 0] = _p4(200, MPI0)
    prot[2, 0] = _p4(700, MP); prot[2, 1] = _p4(600, MP)
    prot[3, 0] = _p4(700, MP); pid[3, 0] = 211; pip[3, 0] = _p4(400, MPI)
    p = tmp / "oracle.npz"
    np.savez(p, lep=np.array([_p4(500, MU)] * 4, float), prot_p4=prot, pi_p4=pip, pi_pid=pid,
             w=np.ones(4), weight_to_nb=1.0, proc=np.array([200, 200, 200, 401]))
    return str(p)


def test_bank_signal_cc0pi_topology(tmp_path):
    b = _write_bank(tmp_path)
    r = SG.bank_signal(b, _CC0PI)
    assert len(r["w"]) == 2, "CC0pi ge1 -> events 0 and 2 (meson-veto drops 1 and 3)"
    assert _KEYS <= set(r), f"missing STV keys: {_KEYS - set(r)}"


def test_bank_signal_proton_count_eq1(tmp_path):
    b = _write_bank(tmp_path)
    assert len(SG.bank_signal(b, _CC0PI_EQ1)["w"]) == 1, "eq1 -> only event 0 (event 2 has 2 in-acc protons)"


def test_bank_signal_cc1pi(tmp_path):
    b = _write_bank(tmp_path)
    r = SG.bank_signal(b, _CC1PI)
    assert len(r["w"]) == 1, "CC1pi+ -> only event 3"
    assert set(r["chan"]) <= {0, 1}


def test_oracle_signal_mirrors_bank(tmp_path):
    o = _write_oracle(tmp_path)
    assert len(SG.oracle_signal(o, _CC0PI)["w"]) == 2       # events 0, 2
    assert len(SG.oracle_signal(o, _CC0PI_EQ1)["w"]) == 1   # event 0
    r1 = SG.oracle_signal(o, _CC1PI)
    assert len(r1["w"]) == 1                                # event 3
    assert int(r1["chan"][0]) == 1                          # proc 401 != 200 -> RES
    assert _KEYS <= set(r1)
