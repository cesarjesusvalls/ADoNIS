"""The fit config must describe the sec4 P1 campaign exactly, and must reject a typo.

The regression this guards: the P1 run's parameters lived only as environment variables spread across
SLURM submitters, and the one that was missing from one script (S4_PRIOR_SCALE, absent from the corner)
made that stage run MAP while every other stage ran MLE.  A config is only an improvement if it is
provably the same numbers -- hence the exact-string check below -- and only safe if a mistyped key is an
error rather than a silent default.
"""
import pytest

yaml = pytest.importorskip("yaml")

from adonis.fit.config import FitConfig

CFG = "configs/fits/sec4_P1.yaml"

# The literal string the campaign ran with, recovered from the submitters.  Do not reformat.
CAMPAIGN_INJECT = (
    "M_A_qe=1.12,M_A_res=0.85,axial_strength=1.1,res_axial_strength=0.8,delta_strength=0.9,"
    "sabs=1.18,s_piN_elastic=0.88,s_conv=1.15,s_NN_elastic[0]=0.82,s_NN_elastic[1]=1.25,"
    "s_NN_elastic[2]=0.9,s_NN_inelastic[0]=1.2,s_NN_inelastic[2]=0.86,f_NN_cex=0.4,kF_sf=1.1,"
    "Eb_shift=0.5,src_tail=0.92"
)


@pytest.fixture(scope="module")
def cfg():
    return FitConfig.load(CFG)


def test_inject_string_matches_the_campaign(cfg):
    """Indexed knobs are written as a list (s_NN_elastic) and as a sparse mapping (s_NN_inelastic, whose
    [1] never moved); both must flatten to the bracket form parse_inject expects, in order."""
    assert cfg.inject_string() == CAMPAIGN_INJECT


def test_the_numbers_that_define_the_run(cfg):
    assert cfg.fit.estimator == "mle" and cfg.fit.prior_scale == 1e6
    assert cfg.fit.minimizer.method == "trf"
    assert cfg.fit.minimizer.max_nfev == 200
    assert cfg.fit.minimizer.gtol == 1e-8
    # xtol/ftol stay tight on purpose: stopping on a small step is how LM false-converged
    assert cfg.fit.minimizer.xtol == 1e-14 and cfg.fit.minimizer.ftol == 1e-14
    assert cfg.data.sigma.syst == 0.05
    assert cfg.data.sigma.mc_term is False          # syst-only; the MC term is a mask input, not a sigma
    assert cfg.data.sigma.mask_mcfrac == 0.05
    assert cfg.banks.sig_cap == 250_000
    assert len(cfg.samples) == 7 and len(cfg.beams) == 3


def test_stages_present_and_shaped(cfg):
    assert [u["method"] for u in cfg.uncertainty] == \
        ["gaussian", "profile", "profile2d", "gradient2d", "nuts", "toys"]
    assert cfg.stage("profile2d")["n"] == 41
    assert cfg.stage("profile2d")["dials"] == \
        ["M_A_res", "delta_strength", "res_axial_strength", "Eb_shift"]
    assert cfg.stage("nuts")["chains"] * cfg.stage("nuts")["samples"] == 24_000
    assert cfg.stage("toys")["n"] == 2000 and cfg.stage("toys")["fixed_truth"] is True
    with pytest.raises(KeyError):
        cfg.stage("no_such_method")


@pytest.mark.parametrize("where,key", [("fit", "estimatr"), ("data", "injekt"), (None, "extra_top")])
def test_unknown_keys_are_rejected(tmp_path, where, key):
    raw = yaml.safe_load(open(CFG))
    (raw if where is None else raw[where])[key] = 1
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown key"):
        FitConfig.load(p)


@pytest.mark.parametrize("patch,msg", [
    ({"fit": {"estimator": "maximum_likelihood"}}, "estimator"),
    ({"fit": {"minimizer": {"method": "levenberg"}}}, "method"),
    ({"data": {"mode": "toy"}}, "mode"),
])
def test_invalid_values_are_rejected(tmp_path, patch, msg):
    raw = yaml.safe_load(open(CFG))
    for k, v in patch.items():
        raw[k].update(v) if not isinstance(list(v.values())[0], dict) else raw[k][list(v)[0]].update(list(v.values())[0])
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=msg):
        FitConfig.load(p)


def test_digest_is_stable_and_content_addressed(cfg, tmp_path):
    """Two loads of the same file agree; a changed number does not.  This is what a stage manifest
    carries so a merged product can prove its shards came from one definition."""
    assert cfg.digest() == FitConfig.load(CFG).digest()
    raw = yaml.safe_load(open(CFG))
    raw["fit"]["minimizer"]["max_nfev"] = 201
    p = tmp_path / "other.yaml"
    p.write_text(yaml.safe_dump(raw))
    assert FitConfig.load(p).digest() != cfg.digest()
