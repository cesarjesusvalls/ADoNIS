"""Fast tests for the section-3 subset resolver (analysis.paper.sec3_fisher.subsets).

The point of the resolver is that sec3 survives a CHANGE of sample composition or binning without a
code edit, so that is what is pinned here: globs that match nothing, groups that vanish, references to
a vanished group, and datasets no column claims.  Pure numpy/yaml on hand-built npz -- no jax, no bank.
"""
import numpy as np
import pytest
import yaml

from analysis.paper.sec3_fisher import subsets as SS

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


# ---- schema contract ---------------------------------------------------------------------------- #

def test_check_schema_accepts_and_returns(tmp_path):
    J, sigma, prior, pnames, dskeys, row0 = SS.check_schema(_npz(tmp_path))
    assert J.shape == (25, 4) and len(dskeys) == 7 and row0[-1] == 25 and len(pnames) == 4


@pytest.mark.parametrize("over, msg", [
    (dict(sigma=np.ones(3)), "sigma"),                       # bins disagree with J
    (dict(prior=np.full(9, 0.2)), "prior/pnames"),           # knobs disagree with J
    (dict(row0=np.array([0, 3, 25])), "offsets"),            # row0 length != len(dskeys)+1
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


# ---- group resolution --------------------------------------------------------------------------- #

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
        ("both", ["pmu", "cosmu", "mnv_dpt", "mnv_pn", "e_qe"]),   # ref order preserved, de-duped
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
    assert [n for n, _l, _k in got] == ["t2k", "all"]          # 'neut' dropped, 'all' survives
    assert got[-1][2] == ["pmu", "cosmu"]                      # <neut> contributed nothing
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


# ---- row slicing -------------------------------------------------------------------------------- #

def test_rows_for_picks_the_right_bins(tmp_path):
    *_, dskeys, row0 = SS.check_schema(_npz(tmp_path))
    assert list(SS.rows_for(["e_qe"], dskeys, row0)) == [14, 15, 16, 17, 18]
    # order-independent, and the union is the sorted concatenation
    assert (SS.rows_for(["pip_react", "pmu"], dskeys, row0)
            == np.array([0, 1, 2, 19, 20, 21, 22, 23, 24])).all()


def test_rows_for_a_rebinned_npz_follows_the_new_edges(tmp_path):
    """Binning is read from row0, never assumed: same keys, different bin counts -> different rows."""
    *_, dskeys, row0 = SS.check_schema(_npz(tmp_path, nbins=[1, 1, 1, 1, 1, 1, 1]))
    assert list(SS.rows_for(["e_qe"], dskeys, row0)) == [5]


def test_config_file_resolves_against_the_shipped_datasets():
    """The committed YAML must parse and produce columns for a T2K-only npz."""
    cfg = SS.load_config()
    t2k = ["dpt", "dat", "pmu", "cosmu", "pn", "dptt", "daT", "ppi", "cospi", "n_p", "n_chpi"]
    axes = {a: SS.resolve_axis(dict(spec, _name=a), t2k, log=lambda *_: None)
            for a, spec in cfg["axes"].items()}
    assert [n for n, _l, _k in axes["obs_classes"]] == ["lepton", "leptonhad", "tki", "mult",
                                                        "kin9", "full"]
    assert len(axes["obs_classes"][-1][2]) == 11                 # full == kin9 + mult
    assert [n for n, _l, _k in axes["samples_alone"]] == ["T2K"]  # the other samples aren't there
    assert all(len(ks) == 11 for _n, _l, ks in axes["sample_ladder"])   # ladder collapses to T2K


def test_obs_classes_is_fed_the_full_t2k_observable_set():
    """obs_classes references all 11 T2K observables (incl. ppi/cospi/n_p/n_chpi), so it MUST read the
    single-sample T2K npz (physfit_gate1), and the sample axes MUST read a `_full` multisample -- NOT the
    sec2-figure `multisample_carbon`, whose keep-filter drops 4 of the 11 and would silently collapse the
    `mult`/`kin9`/`full` columns and understate the T2K rung.  Regression guard for the sec1/2/3 merge."""
    cfg = SS.load_config()
    assert cfg["axes"]["obs_classes"].get("npz") == "physfit_gate1"
    assert cfg.get("npz", "").endswith("_full")
