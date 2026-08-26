"""Typed unfolding configuration, loaded from `configs/fits/*.yaml`.

One object describes a run: which sample, what detector, which binning, what priors, which studies.
Nothing in the unfolding path reads the environment.  Unknown keys are an error, not a warning.

    cfg = UnfoldConfig.load("configs/fits/<study>.yaml")
    cfg.smear_spec()      # SmearSpec for the detector block
    cfg.grids()           # (truth_grid, reco_grid)
    cfg.as_dict()         # resolved config, stamped into the npz
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


def _only(d: dict, allowed: set, where: str) -> dict:
    """Reject unknown keys.  A typo in a config must fail loudly, not fall back to a default."""
    extra = set(d) - allowed
    if extra:
        raise ValueError(f"{where}: unknown key(s) {sorted(extra)}; allowed {sorted(allowed)}")
    return d


@dataclass(frozen=True)
class Detector:
    """The smearing model.  `pi_eff` matters beyond realism: at pi_eff=1 (perfect PID) a topological
    CC0pi sample has no background, so knobs that only reweight the background have nothing to act
    on."""
    sigma_p: float = 0.10
    sigma_theta_deg: float = 5.0
    pi_eff_p_max: float = 400.0
    pi_eff: float = 0.5
    seed: int = 20260813


@dataclass(frozen=True)
class Binning:
    """observables: "stv" (delta-p_T, delta-alpha_T) or "lep" (p_mu, cos theta_mu).

    `truth` names the rectangular variant for "stv" and is ignored for "lep", which always uses the
    published T2K staircase.  `reco_split` subdivides each staircase p_mu bin: templates carry no
    prior, so unfolding needs strictly more reco bins than truth bins."""
    observables: str = "lep"
    truth: str = "3x3"
    reco_split: int = 2


@dataclass(frozen=True)
class Flux:
    """Per-bin prior width and the correlation length of the exponential kernel.  The off-diagonal
    terms keep the fit from absorbing a single reco bin's fluctuation into a sawtooth-shaped flux.
    `edges: null` keeps the module default."""
    sigma: float = 0.10
    corr_length: float = 400.0
    edges: list | None = None


@dataclass(frozen=True)
class Priors:
    """detector: per-reco-bin normalisation width.  dial_threshold: a cross-section knob floats when a
    prior-sized move of it shifts the background by >= this, in units of the Poisson error of the total
    predicted spectrum, L2-summed over reco bins."""
    detector: float = 0.05
    dial_threshold: float = 1.0


@dataclass(frozen=True)
class Study:
    """One fake-data throw.  `scale_c` scales every template; `scale_flux_peak` scales the flux in
    the peak bins instead, so the two can be used to test separability between them."""
    name: str
    scale_c: float = 1.0
    scale_flux_peak: float = 1.0
    flux_tilt: float = 0.0


@dataclass(frozen=True)
class Figures:
    """Figure-only choices, stamped into the npz so the plot scripts stay pure npz consumers."""
    slices: list = field(default_factory=lambda: [[0.20, 0.60], [0.94, 0.98]])
    aspect: float = 0.5


@dataclass(frozen=True)
class UnfoldConfig:
    name: str
    sample: str
    norm_events: int = 200_000
    max_chunks: int | None = None
    detector: Detector = field(default_factory=Detector)
    binning: Binning = field(default_factory=Binning)
    flux: Flux = field(default_factory=Flux)
    priors: Priors = field(default_factory=Priors)
    studies: tuple = ()
    budget: tuple = ("stat", "xsec", "flux", "det")
    figures: Figures = field(default_factory=Figures)

    @classmethod
    def load(cls, path):
        d = yaml.safe_load(Path(path).read_text()) or {}
        _only(d, {"name", "sample", "norm_events", "max_chunks", "detector", "binning", "flux",
                  "priors", "studies", "budget", "figures"}, str(path))
        sub = {}
        for key, kind in (("detector", Detector), ("binning", Binning), ("flux", Flux),
                          ("priors", Priors), ("figures", Figures)):
            blk = d.get(key) or {}
            _only(blk, set(kind.__dataclass_fields__), f"{path}:{key}")
            sub[key] = kind(**blk)
        studies = []
        for st in d.get("studies") or [{"name": "asimov"}]:
            _only(st, set(Study.__dataclass_fields__), f"{path}:studies")
            studies.append(Study(**st))
        return cls(name=d["name"], sample=d["sample"],
                   norm_events=int(d.get("norm_events", 200_000)),
                   max_chunks=d.get("max_chunks"),
                   studies=tuple(studies), budget=tuple(d.get("budget", cls.budget)), **sub)

    def smear_spec(self):
        from adonis.detector import SmearSpec
        return SmearSpec(**asdict(self.detector))

    def grids(self):
        from adonis.unfold.binning import reco_grid, truth_grid
        o = self.binning.observables
        return truth_grid(o, self.binning.truth), reco_grid(o, self.binning.reco_split)

    def sample_path(self):
        """`sample` may be a bare name or a path; a bare name resolves under configs/samples/."""
        p = Path(self.sample)
        return str(p if p.suffix == ".yaml" else Path("configs/samples") / f"{self.sample}.yaml")

    def as_dict(self):
        """The RESOLVED configuration, for stamping into the npz.  A figure or a reader can then state
        the fit it came from without consulting a yaml that may since have changed."""
        d = asdict(self)
        d["studies"] = [asdict(s) if not isinstance(s, dict) else s for s in self.studies]
        return d

    def as_json(self):
        return json.dumps(self.as_dict(), sort_keys=True, default=str)
