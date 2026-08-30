"""A merge must refuse an incomplete or inconsistent shard set.

Each case below is one where a wrong figure is the alternative to an error:

  * a missing row block, which would drop the fine scan and draw the coarse one in its place;
  * two campaigns under one label, whose axes agree while their estimators do not;
  * a preempted shard, whose NaNs are indistinguishable from a genuinely undefined region.

The checks are exercised on synthetic npz files: the real products predate stamping (that path is tested
too), and reproducing a corner shard means running the fit.
"""
import numpy as np
import pytest

from adonis.fit import merge as MG
from adonis.fit import provenance


def _shard(tmp_path, name, *, digest="abc123", rows=None, grid=21, complete=True, stamped=True):
    p = tmp_path / f"{name}.npz"
    kw = {}
    if stamped:
        kw.update({"prov_digest": digest, "prov_config": "sec4_P1.yaml", "prov_git": "deadbee",
                   "prov_stage": "profile2d"})
        if rows is not None:
            kw.update({"prov_row_base": rows[0], "prov_n_row": rows[1], "prov_n_grid": grid})
    np.savez(p, complete=complete, chi2_abs=np.zeros((grid, grid)), **kw)
    return str(p)


def _load(files):
    return [np.load(f, allow_pickle=True) for f in files]



def test_stamp_reads_back(tmp_path):
    f = _shard(tmp_path, "s", rows=(0, 7))
    p = provenance.read(np.load(f, allow_pickle=True))
    assert p["digest"] == "abc123" and p["row_base"] == 0 and p["n_row"] == 7


def test_unstamped_reads_as_none(tmp_path):
    f = _shard(tmp_path, "old", stamped=False)
    assert provenance.read(np.load(f, allow_pickle=True)) is None


def test_stamp_survives_no_config_in_env(monkeypatch):
    """Run outside the runner: the digest is empty but the commit is still recorded, and nothing raises."""
    monkeypatch.delenv("ADONIS_FIT_CONFIG", raising=False)
    s = provenance.stamp(row_base=3)
    assert s["prov_digest"] == "" and s["prov_row_base"] == 3 and s["prov_git"]



def test_one_run_definition_passes(tmp_path):
    fs = [_shard(tmp_path, f"s{i}", rows=(7 * i, 7)) for i in range(3)]
    rep = MG.check(fs, _load(fs))
    assert rep.ok and rep.digest == "abc123" and not rep.warnings


def test_two_run_definitions_is_an_error(tmp_path):
    """The MAP-vs-MLE shape: same axes, different run.  Must not merge."""
    fs = [_shard(tmp_path, "a", rows=(0, 7)), _shard(tmp_path, "b", rows=(7, 7), digest="999zzz")]
    rep = MG.check(fs, _load(fs))
    assert not rep.ok
    assert "different run definitions" in rep.errors[0]
    with pytest.raises(SystemExit, match="refusing to merge"):
        rep.raise_if_bad()


def test_unfinished_shard_is_an_error(tmp_path):
    fs = [_shard(tmp_path, "a", rows=(0, 7)), _shard(tmp_path, "b", rows=(7, 7), complete=False)]
    rep = MG.check(fs, _load(fs))
    assert not rep.ok and "did not finish" in " ".join(rep.errors)


def test_unstamped_shards_warn_but_do_not_block(tmp_path):
    """A shard with no provenance stamp must warn, not block: a check that always refused unstamped
    input could never process data written before the stamp existed."""
    fs = [_shard(tmp_path, f"s{i}", stamped=False) for i in range(3)]
    rep = MG.check(fs, _load(fs))
    assert rep.ok and rep.warnings and "no provenance" in rep.warnings[0]
    rep.raise_if_bad()



def test_row_blocks_tiling_the_grid_pass():
    rep = MG.Report("x")
    assert rep.rows([(0, 7), (7, 7), (14, 7)], grid=21) and rep.ok


def test_missing_rows_are_named_not_counted():
    """The point of the whole exercise: say WHICH rows, so the fix is obvious."""
    rep = MG.Report("x")
    rep.rows([(0, 7), (14, 7)], grid=21)
    assert not rep.ok and "rows not computed: 7-13 of 21" in rep.errors[0]


def test_a_single_missing_row_still_fails():
    """A single missing grid row must fail the check, however small a fraction of the total it is."""
    rep = MG.Report("x")
    rep.rows([(0, 20)], grid=21)
    assert not rep.ok and "20" in rep.errors[0]


def test_overlapping_blocks_warn():
    rep = MG.Report("x")
    rep.rows([(0, 10), (7, 14)], grid=21)
    assert rep.ok and "more than one shard" in rep.warnings[0]


def test_rows_past_the_end_are_an_error():
    rep = MG.Report("x")
    rep.rows([(0, 7), (7, 21)], grid=21)
    assert not rep.ok and "outside the grid" in " ".join(rep.errors)


@pytest.mark.parametrize("xs,want", [([0, 1, 2, 7], "0-2,7"), ([5], "5"), ([1, 3, 5], "1,3,5"),
                                     (list(range(300)), "0-299")])
def test_run_compression(xs, want):
    assert MG._runs(xs) == want



def test_allow_partial_merges_but_marks_the_product(tmp_path):
    fs = [_shard(tmp_path, "a", rows=(0, 7), complete=False)]
    rep = MG.check(fs, _load(fs))
    rep.raise_if_bad(allow_partial=True)
    assert rep.stamp()["prov_partial"] is True


def test_clean_merge_is_not_marked_partial(tmp_path):
    fs = [_shard(tmp_path, f"s{i}", rows=(7 * i, 7)) for i in range(3)]
    rep = MG.check(fs, _load(fs))
    st = rep.stamp()
    assert st["prov_partial"] is False and st["prov_digest"] == "abc123" and st["prov_git"] == "deadbee"


def test_emit_does_not_repeat_itself(tmp_path, capsys):
    """A merge validates in stages, so emit() runs more than once; the user must not see doubles."""
    fs = [_shard(tmp_path, "s", stamped=False)]
    rep = MG.check(fs, _load(fs))
    rep.emit(); rep.emit()
    assert capsys.readouterr().out.count("no provenance") == 1



def test_allow_partial_registers_a_degradation(tmp_path):
    """A figure is a PNG with nowhere to keep prov_partial, so the fact is registered process-wide for
    the renderer to find -- without every figure function having to remember to pass a flag."""
    MG.DEGRADED.clear()
    fs = [_shard(tmp_path, "a", rows=(0, 7), complete=False)]
    MG.check(fs, _load(fs), what="thing").raise_if_bad(allow_partial=True)
    assert len(MG.DEGRADED) == 1 and MG.DEGRADED[0][0] == "thing"
    MG.DEGRADED.clear()


def test_clean_run_registers_nothing(tmp_path):
    MG.DEGRADED.clear()
    fs = [_shard(tmp_path, f"s{i}", rows=(7 * i, 7)) for i in range(3)]
    MG.check(fs, _load(fs)).raise_if_bad()
    assert MG.DEGRADED == []


def test_figure_is_stamped_when_partial(tmp_path, monkeypatch):
    """End to end: the banner is drawn and the reason lands in the PNG metadata."""
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from analysis.paper import style

    monkeypatch.setattr(style, "OUTDIR", tmp_path)
    MG.DEGRADED.clear()
    MG.DEGRADED.append(("sec4_P1 corner2d", "falling back to N=21"))
    try:
        style.save(plt.figure(), "figtest")
        img = Image.open(tmp_path / "figtest.png")
        assert "falling back to N=21" in img.text["Comment"]
    finally:
        MG.DEGRADED.clear()


def test_figure_is_not_stamped_when_clean(tmp_path, monkeypatch):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from analysis.paper import style

    monkeypatch.setattr(style, "OUTDIR", tmp_path)
    MG.DEGRADED.clear()
    style.save(plt.figure(), "figclean")
    assert Image.open(tmp_path / "figclean.png").text["Comment"] == "complete"
