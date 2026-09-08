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


def test_probe_travels_as_structure_not_value():
    """to_jax carries the EM probe as a key, so it stays static under jit."""
    import numpy as np
    from adonis.reweight.bank_reweight import to_jax
    base = dict(w0=np.ones(3), channel=np.zeros(3, np.int32), p_struck=np.zeros((3, 4)),
                hv_qe_mij=np.zeros((3, 4, 4)), hv_qe_Q2=np.zeros(3))
    em = to_jax({**base, "hv_qe_probe_em": np.int32(1)})
    cc = to_jax({**base, "hv_qe_probe_em": np.int32(0)})
    plain = to_jax(base)
    assert "hv_qe_em" in em and "hv_qe_em" not in cc and "hv_qe_em" not in plain
    for d in (em, cc, plain):
        assert "hv_qe_probe_em" not in d


def test_probe_is_not_read_as_a_value():
    """A jitted bank dict makes every value a tracer, so the probe must come from key presence."""
    import inspect
    from adonis.reweight import bank_reweight
    src = inspect.getsource(bank_reweight.bank_weight)
    assert '"hv_qe_em" in B' in src
    assert "hv_qe_probe_em" not in src


def test_jitted_bank_dict_keeps_the_probe_static():
    import jax, jax.numpy as jnp, numpy as np
    from adonis.reweight.bank_reweight import to_jax
    B = to_jax(dict(w0=np.ones(3), channel=np.zeros(3, np.int32), p_struck=np.zeros((3, 4)),
                    hv_qe_mij=np.zeros((3, 4, 4)), hv_qe_Q2=np.zeros(3),
                    hv_qe_probe_em=np.int32(1)))

    @jax.jit
    def probe_of(bank):                       # mirrors how bank_weight is jitted with the bank as an arg
        return jnp.float64(1.0) if "hv_qe_em" in bank else jnp.float64(0.0)

    assert float(probe_of(B)) == 1.0
