"""Fast tests for the section-3 subset resolver (analysis.paper.grad_info.subsets).

The point of the resolver is that sec3 survives a CHANGE of sample composition or binning without a
code edit, so that is what is pinned here: globs that match nothing, groups that vanish, references to
a vanished group, and datasets no column claims.  Pure numpy/yaml on hand-built npz -- no jax, no bank.
"""
import numpy as np
import pytest
import yaml

from analysis.paper.grad_info import subsets as SS

DSKEYS = ["pmu", "cosmu", "dpt", "mnv_dpt", "mnv_pn", "e_qe", "pip_react"]
NBINS = [3, 3, 4, 2, 2, 5, 6]


def _npz(tmp_path, dskeys=DSKEYS, nbins=NBINS, nknob=4, **over):
    row0 = np.concatenate([[0], np.cumsum(nbins)])
    d = dict(J=np.ones((row0[-1], nknob)), sigma=np.ones(row0[-1]), prior=np.full(nknob, 0.2),
             pnames=np.array([f"k{i}" for i in range(nknob)]), dskeys=np.array(dskeys), row0=row0)
    d.update(over)
    f = tmp_path / "j.npz"
    np.savez(f, **d)
    return np.load(f, allow_pickle=True)


def _axis(groups, name="ax"):
    return dict(yaml.safe_load(groups), _name=name)


def _declared_dskeys():
    """The real observable stack, read from the SAME declarations the Jacobian builder reads.

    Hardcoding it is what broke this test: the keys became namespaced `sample:obs` (commit a41b9b1) and
    the literal list here stayed on the old bare names, so `t2k_*`/`ee_*` matched nothing and two probe
    groups silently vanished from an assertion that was still checking group NAMES.  Deriving them means
    a change of sample composition -- exactly what the resolver exists to survive -- updates this test
    instead of breaking it.
    """
    import pathlib

    from adonis.fit.config import FitConfig
    cfg = FitConfig.load("configs/fits/sec4_P1.yaml")
    beams = yaml.safe_load(pathlib.Path("configs/samples/beams.yaml").read_text())
    out = []
    for nm in cfg.samples:
        s = yaml.safe_load(pathlib.Path(f"configs/samples/{nm}.yaml").read_text())
        out += [f"{s['name']}:{o['key']}" for o in s["observables"] if o.get("fit", True)]
    for b in cfg.beams:
        out += list(beams["beams"][b]["observables"])
    return out



def test_check_schema_accepts_and_returns(tmp_path):
    J, sigma, prior, pnames, dskeys, row0 = SS.check_schema(_npz(tmp_path))
    assert J.shape == (25, 4) and len(dskeys) == 7 and row0[-1] == 25 and len(pnames) == 4


@pytest.mark.parametrize("over, msg", [
    (dict(sigma=np.ones(3)), "sigma"),
    (dict(prior=np.full(9, 0.2)), "prior/pnames"),
    (dict(row0=np.array([0, 3, 25])), "offsets"),
])
def test_check_schema_rejects_inconsistent(tmp_path, over, msg):
    with pytest.raises((ValueError, KeyError), match=msg):
        SS.check_schema(_npz(tmp_path, **over), "bad")


def test_check_schema_rejects_non_physfit_npz(tmp_path):
    f = tmp_path / "other.npz"
    np.savez(f, something_else=np.zeros(3))
    with pytest.raises(KeyError, match="physfit-schema"):
        SS.check_schema(np.load(f), "other")


def test_row0_must_span_J(tmp_path):
    with pytest.raises(ValueError, match="row0 spans"):
        SS.check_schema(_npz(tmp_path, row0=np.array([0, 3, 6, 10, 12, 14, 19, 24])), "short")



def test_literals_globs_and_refs():
    got = SS.resolve_axis(_axis("""
        groups:
          lepton: {keys: [pmu, cosmu]}
          mnv:    {label: MINERvA, keys: ["mnv_*"]}
          both:   {keys: ["<lepton>", "<mnv>", "e_*"]}
    """), DSKEYS, log=lambda *_: None)
    assert [(n, ks) for n, _l, ks in got] == [
        ("lepton", ["pmu", "cosmu"]),
        ("mnv", ["mnv_dpt", "mnv_pn"]),
        ("both", ["pmu", "cosmu", "mnv_dpt", "mnv_pn", "e_qe"]),
    ]
    assert got[1][1] == "MINERvA"


def test_absent_sample_drops_its_column_but_keeps_the_rest():
    """The whole point: an npz WITHOUT the hadron beams must still render every other column."""
    logs = []
    got = SS.resolve_axis(_axis("""
        groups:
          t2k:  {keys: [pmu, cosmu]}
          neut: {keys: ["neut_*"]}
          all:  {keys: ["<t2k>", "<neut>"]}
    """), ["pmu", "cosmu"], log=logs.append)
    assert [n for n, _l, _k in got] == ["t2k", "all"]
    assert got[-1][2] == ["pmu", "cosmu"]
    assert any("DROPPED" in m for m in logs)


def test_reference_to_undeclared_group_is_an_error():
    with pytest.raises(KeyError, match="not an earlier group"):
        SS.resolve_axis(_axis("groups: {a: {keys: ['<later>']}, later: {keys: [pmu]}}"),
                        DSKEYS, log=lambda *_: None)


def test_uncovered_datasets_are_reported_not_silently_dropped():
    logs = []
    SS.resolve_axis(_axis("groups: {lepton: {keys: [pmu, cosmu]}}"), DSKEYS, log=logs.append)
    assert any("UNCOVERED" in m and "pip_react" in m for m in logs)


def test_bare_list_group_is_accepted():
    got = SS.resolve_axis(_axis("groups: {lepton: [pmu, cosmu]}"), DSKEYS, log=lambda *_: None)
    assert got[0][2] == ["pmu", "cosmu"]



def test_rows_for_picks_the_right_bins(tmp_path):
    *_, dskeys, row0 = SS.check_schema(_npz(tmp_path))
    assert list(SS.rows_for(["e_qe"], dskeys, row0)) == [14, 15, 16, 17, 18]
    assert (SS.rows_for(["pip_react", "pmu"], dskeys, row0)
            == np.array([0, 1, 2, 19, 20, 21, 22, 23, 24])).all()


def test_rows_for_a_rebinned_npz_follows_the_new_edges(tmp_path):
    """Binning is read from row0, never assumed: same keys, different bin counts -> different rows."""
    *_, dskeys, row0 = SS.check_schema(_npz(tmp_path, nbins=[1, 1, 1, 1, 1, 1, 1]))
    assert list(SS.rows_for(["e_qe"], dskeys, row0)) == [5]


def test_config_file_resolves_against_the_multisample_datasets():
    """The committed YAML parses and groups the real observable stack by probe -- T2K + MINERvA, the
    (e,e') sample, and the pi+/p/n beams -- with every dataset claimed exactly once."""
    cfg = SS.load_config()
    dskeys = _declared_dskeys()
    axes = {a: SS.resolve_axis(dict(spec, _name=a), dskeys, log=lambda *_: None)
            for a, spec in cfg["axes"].items()}
    assert list(cfg["axes"]) == ["by_probe"]
    assert [n for n, _l, _k in axes["by_probe"]] == ["nu", "ebeam", "hadr", "all"]
    keys = {n: k for n, _l, k in axes["by_probe"]}
    assert set(keys["nu"]) | set(keys["ebeam"]) | set(keys["hadr"]) == set(dskeys)
    assert len(keys["nu"]) + len(keys["ebeam"]) + len(keys["hadr"]) == len(dskeys)
    assert set(keys["all"]) == set(dskeys)
    assert all(k.startswith(("t2k_", "minerva_")) for k in keys["nu"])
    assert all(k.startswith("ee_") for k in keys["ebeam"])


def test_all_axes_read_the_shared_multisample_npz():
    """Every sec2 axis reads the SAME multisample_carbon npz -- the same samples + signal defs the sec1
    validation figures use (T2K keep-filtered to 7).  No per-axis npz override remains after aligning
    sec2/sec3 to sec2's samples.  Regression guard for the swap."""
    cfg = SS.load_config()
    assert cfg.get("npz") == "multisample_carbon"
    assert all("npz" not in spec for spec in cfg["axes"].values())
