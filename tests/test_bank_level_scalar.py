"""A bank-level scalar field survives a load/pack round trip.

The EM/CC probe flag is stored once per chunk, not once per event; concatenating it across chunks
raises, and dropping it makes an electron bank reweight as if it were a neutrino one.
"""
import numpy as np
import pytest

from adonis.reweight.bank_plot import load_bank, filter_events


def _chunk(path, n, em):
    np.savez(path, w0=np.full(n, 0.5), channel=np.zeros(n, np.int32),
             fs_off=np.arange(n + 1, dtype=np.int64), fs_pid=np.full(n, 2212, np.int32),
             fs_chg=np.ones(n, np.int32), fs_p4=np.zeros((n, 4)),
             hv_qe_probe_em=np.int32(em))


def _bank(tmp_path, em0, em1):
    d = tmp_path / "bank"; d.mkdir()
    _chunk(d / "chunk_000.npz", 4, em0)
    _chunk(d / "chunk_001.npz", 3, em1)
    (d / "manifest.json").write_text('{"n_chunks": 2}')
    return str(d)


def test_scalar_survives_load():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as t:
        B = load_bank(_bank(pathlib.Path(t), 1, 1))
    assert len(B["w0"]) == 7
    assert np.asarray(B["hv_qe_probe_em"]).ndim == 0
    assert int(np.asarray(B["hv_qe_probe_em"])) == 1


def test_scalar_survives_filter():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as t:
        B = load_bank(_bank(pathlib.Path(t), 1, 1))
    keep = np.zeros(7, bool); keep[[0, 3, 5]] = True
    out = filter_events(B, keep)
    assert len(out["w0"]) == 3
    assert int(np.asarray(out["hv_qe_probe_em"])) == 1


def test_disagreeing_chunks_raise():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as t:
        with pytest.raises(ValueError, match="bank-level"):
            load_bank(_bank(pathlib.Path(t), 1, 0))
