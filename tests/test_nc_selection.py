"""G8 -- the NC1pi0 selection path, and the two config lies it fixes.

The NC selection is deliberately a PARALLEL path, not a flag on the CC one.  Every observable `_obs`
computes -- dpt, dalphat, dphit, pn, dptt -- takes the outgoing lepton as a required argument and is
physically undefined without it.  For NC the lepton is an invisible neutrino, so a "lepton optional"
flag would yield observables that are silently meaningless rather than absent.

The sharpest test here is the veto test.  `n_other_meson` counts pi0 and pi- as "other mesons"
(extract.py: `n_other += (pid != PIP)`), and selection.py then ADDS it to the pion count.  Both terms
are zero on a CC0pi signal, so no CC number was ever wrong -- but reusing that field as a veto for a
pi0 signal vetoes the signal itself, 100% of it, and the figure would come out empty with no error.
"""
import numpy as np
import pytest

from adonis.workflow.config import NuSignalDef
from adonis.reweight import bank_plot as BP


def _bank(pids_per_event, w=None):
    """Minimal ragged bank: pids_per_event is a list of final-state PID lists."""
    n = len(pids_per_event)
    flat = [p for ev in pids_per_event for p in ev]
    eidx = np.concatenate([np.full(len(ev), i) for i, ev in enumerate(pids_per_event)]).astype(int) \
        if flat else np.zeros(0, int)
    p4 = np.zeros((len(flat), 4))
    for i, p in enumerate(flat):
        p4[i] = [np.sqrt(300.0 ** 2 + 140.0 ** 2), 0.0, 60.0, 294.0]
    return {"w0": np.ones(n) if w is None else np.asarray(w),
            "fs_pid": np.asarray(flat, int), "fs_p4": p4, "_eidx": eidx,
            "channel": np.ones(n, np.int8),
            "k_lep": np.tile([1000.0, 0.0, 100.0, 990.0], (n, 1))}


def _sd(**kw):
    """Wide-open acceptance: these tests are about TOPOLOGY (which pions/mesons pass the veto), so
    every kinematic window is opened out.  A window that quietly rejects the fixture would make the
    tests pass vacuously with zero events on both sides."""
    d = dict(pi_win=(0.0, 1e9), p_win=(0.0, 1e9), mu_win=(0.0, 1e9),
             cth=-1.0, cos_mu=-1.0, proton_count="ge1")
    d.update(kw)
    return NuSignalDef(**d)


def _bank_dir(tmp_path, B, monkeypatch):
    """A one-chunk bank directory `_stream_select` will stream: it reads the manifest for the w0
    divisor, then globs chunk_*.npz and hands each to the loader."""
    from adonis.workflow import selection as S
    (tmp_path / "manifest.json").write_text('{"n_chunks": 1}')
    (tmp_path / "chunk_0.npz").write_bytes(b"")
    monkeypatch.setattr(S.BP, "load_bank_chunk", lambda f, nch: B)
    return str(tmp_path)


def test_single_pi0_mirrors_single_pip():
    B = _bank([[111, 2212], [211, 2212]])
    pi0 = BP.single_pi0(B); pip = BP.single_pip(B)
    assert np.any(pi0[0] != 0) and np.all(pi0[1] == 0), "single_pi0 picked the wrong event"
    assert np.any(pip[1] != 0) and np.all(pip[0] == 0), "single_pip changed behaviour"


def test_nc_selection_keeps_one_pi0_and_rejects_charged_pions(tmp_path, monkeypatch):
    from adonis.workflow import selection as S
    B = _bank([[111, 2212],
               [111, 111, 2212],
               [111, 211, 2212],
               [2212]])
    bank = _bank_dir(tmp_path, B, monkeypatch)
    out = S.bank_signal_nc(bank, _sd())
    assert len(out["w"]) == 1, f"expected exactly the 1-pi0 event, got {len(out['w'])}"
    assert out["p_pi0"][0] > 0, "the selected event carries no pion momentum"


def test_a_heavy_meson_vetoes_but_a_pi0_does_not(tmp_path, monkeypatch):
    """THE veto test.  A 1pi0 + 1eta event must be rejected; a plain 1pi0 event must survive.  If
    n_other_meson were reused as the veto, BOTH would be rejected -- the signal along with it."""
    from adonis.workflow import selection as S
    B = _bank([[111, 2212],
               [111, 221, 2212]])
    bank = _bank_dir(tmp_path, B, monkeypatch)
    out = S.bank_signal_nc(bank, _sd())
    assert len(out["w"]) == 1, "the eta event was not vetoed, or the pi0 event was"


def test_nc_selection_never_reads_the_lepton(tmp_path, monkeypatch):
    """G8(4): the NC path must be independent of k_lep entirely.  Zeroing it must change nothing."""
    from adonis.workflow import selection as S
    B = _bank([[111, 2212], [111, 2112]])
    B["k_lep"] = np.tile([1000.0, 0.0, 0.0, 1000.0], (2, 1))
    bank = _bank_dir(tmp_path, B, monkeypatch)
    a = S.bank_signal_nc(bank, _sd())
    B["k_lep"] = np.zeros((2, 4))
    b = S.bank_signal_nc(bank, _sd())
    for k in a:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), f"{k} depended on the lepton"


def test_nc_observables_are_pion_based_and_carry_no_tki():
    """NC has no lepton, so the TKI/STV observables must be ABSENT rather than present-and-zero."""
    from adonis.workflow.selection import _obs_nc
    pi0 = np.array([[400.0, 100.0, 0.0, 350.0]])
    lead = np.array([[1000.0, 0.0, 50.0, 300.0]])
    o = _obs_nc(pi0, lead, np.array([True]))
    assert set(o) == {"p_pi0", "cos_pi0", "th_pi0", "lp_p", "cos_lp", "has_proton"}
    for absent in ("dpt", "dalphat", "dphit", "pn", "dptt", "pmu", "cos_mu"):
        assert absent not in o, f"{absent} is undefined without a lepton but was produced anyway"


def test_anypi_now_differs_from_pip(tmp_path, monkeypatch):
    """`anypi` validated and then did NOTHING -- selection never branched on it, so it was
    byte-identical to `pip`.  A config key that silently does nothing is the same class of defect as
    a field name that lies."""
    from adonis.workflow import selection as S
    B = _bank([[211, 2212], [111, 2212], [-211, 2212]])
    bank = _bank_dir(tmp_path, B, monkeypatch)
    pip = S.bank_signal(bank, _sd(pion_id="pip"))
    anypi = S.bank_signal(bank, _sd(pion_id="anypi"))
    assert len(pip["w"]) == 1, f"the pi+ fixture event should pass `pip`, got {len(pip['w'])}"
    assert len(anypi["w"]) == 3, f"`anypi` should take all three charges, got {len(anypi['w'])}"


def test_pion_id_pi0_is_rejected_by_the_cc_path(tmp_path, monkeypatch):
    """`pi0` is a valid NuSignalDef value, but the CC path must not pretend to handle it -- the NC
    selection is a separate function on purpose, and silently treating pi0 as pi+ is exactly the
    class of defect `anypi` was."""
    from adonis.workflow import selection as S
    B = _bank([[111, 2212]])
    bank = _bank_dir(tmp_path, B, monkeypatch)
    with pytest.raises(ValueError, match="no selection branch"):
        S.bank_signal(bank, _sd(pion_id="pi0"))
