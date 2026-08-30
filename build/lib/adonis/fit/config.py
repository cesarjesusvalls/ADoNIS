"""Typed fit configuration, loaded from YAML.

One object describes a run: which samples, what data, how to fit it, and which uncertainties to
compute.  Everything a stage needs comes from here; no stage reads the environment.  Unknown keys
are an error, not a warning.

    cfg = FitConfig.load("<study>.yaml")
    cfg.inject_string()        # the flat inject-string form parse_inject expects
    cfg.stage("profile2d")     # one uncertainty block by name
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _only(d: dict, allowed: set, where: str) -> dict:
    """Reject unknown keys.  A typo in a config must fail loudly, not fall back to a default."""
    extra = set(d) - allowed
    if extra:
        raise ValueError(f"{where}: unknown key(s) {sorted(extra)}; allowed {sorted(allowed)}")
    return d


@dataclass(frozen=True)
class Banks:
    """Chunk caps: how many events are resident, which sets the MC error per bin and hence the
    sparse-bin mask.  Part of the physics definition, not a performance knob."""
    sig_cap: int = 250_000
    nu_chunks: int = 4
    e_chunks: int = 64
    beam_chunks: int = 5

    @classmethod
    def parse(cls, d):
        return cls(**_only(d or {}, {"sig_cap", "nu_chunks", "e_chunks", "beam_chunks"}, "banks"))


@dataclass(frozen=True)
class Sigma:
    syst: float = 0.05
    mc_term: bool = False
    mask_mcfrac: float = 0.05
    freeze_at: str = "nominal"

    @classmethod
    def parse(cls, d):
        d = _only(d or {}, {"syst", "mc_term", "mask_mcfrac", "freeze_at"}, "data.sigma")
        s = cls(**d)
        if s.freeze_at not in ("nominal", "truth"):
            raise ValueError(f"data.sigma.freeze_at: expected nominal|truth, got {s.freeze_at!r}")
        return s


@dataclass(frozen=True)
class Data:
    mode: str = "closure"
    inject: dict = field(default_factory=dict)
    sigma: Sigma = field(default_factory=Sigma)

    @classmethod
    def parse(cls, d):
        d = _only(dict(d or {}), {"mode", "inject", "sigma"}, "data")
        mode = d.get("mode", "closure")
        if mode not in ("closure", "real"):
            raise ValueError(f"data.mode: expected closure|real, got {mode!r}")
        if mode == "closure" and not d.get("inject"):
            raise ValueError("data.mode=closure requires data.inject (the truth the Asimov data is built at)")
        return cls(mode=mode, inject=dict(d.get("inject") or {}), sigma=Sigma.parse(d.get("sigma")))


@dataclass(frozen=True)
class Minimizer:
    method: str = "trf"
    max_nfev: int = 200
    gtol: float = 1e-8
    xtol: float = 1e-14
    ftol: float = 1e-14
    x_scale: str = "jac"
    start: str = "nominal"
    newton_tol: float = 1e-6

    @classmethod
    def parse(cls, d):
        d = _only(d or {}, {"method", "max_nfev", "gtol", "xtol", "ftol", "x_scale", "start",
                            "newton_tol"},
                  "fit.minimizer")
        m = cls(**d)
        if m.method not in ("trf", "lm"):
            raise ValueError(f"fit.minimizer.method: expected trf|lm, got {m.method!r}")
        if m.start not in ("nominal", "truth"):
            raise ValueError(f"fit.minimizer.start: expected nominal|truth, got {m.start!r}")
        return m


@dataclass(frozen=True)
class Fit:
    dials: Any = "constrained"
    estimator: str = "mle"
    minimizer: Minimizer = field(default_factory=Minimizer)

    @classmethod
    def parse(cls, d):
        d = _only(dict(d or {}), {"dials", "estimator", "minimizer"}, "fit")
        est = d.get("estimator", "mle")
        if est not in ("mle", "map"):
            raise ValueError(f"fit.estimator: expected mle|map, got {est!r}")
        return cls(dials=d.get("dials", "constrained"), estimator=est,
                   minimizer=Minimizer.parse(d.get("minimizer")))

    @property
    def prior_scale(self) -> float:
        """What multiplies the prior width.  MLE is effectively 'no prior': the width is scaled by
        1e6 so the prior term never competes with the data."""
        return 1e6 if self.estimator == "mle" else 1.0


STAGES = {"gaussian", "profile", "profile2d", "gradient2d", "nuts", "toys"}


@dataclass(frozen=True)
class FitConfig:
    name: str
    samples: tuple
    beams: tuple
    banks: Banks
    data: Data
    fit: Fit
    uncertainty: tuple
    compute: dict = field(default_factory=dict)
    path: Path = None

    @classmethod
    def load(cls, path) -> "FitConfig":
        p = Path(path)
        raw = yaml.safe_load(p.read_text()) or {}
        _only(raw, {"name", "samples", "beams", "banks", "data", "fit", "uncertainty", "compute"}, p.name)
        unc = tuple(dict(u) for u in (raw.get("uncertainty") or []))
        for u in unc:
            if u.get("method") not in STAGES:
                raise ValueError(f"uncertainty: unknown method {u.get('method')!r}; "
                                 f"expected one of {sorted(STAGES)}")
        return cls(name=raw["name"],
                   samples=tuple(raw.get("samples") or ()),
                   beams=tuple(raw.get("beams") or ()),
                   compute=dict(raw.get("compute") or {}),
                   banks=Banks.parse(raw.get("banks")),
                   data=Data.parse(raw.get("data")),
                   fit=Fit.parse(raw.get("fit")),
                   uncertainty=unc, path=p)

    def stage(self, method: str) -> dict:
        for u in self.uncertainty:
            if u["method"] == method:
                return u
        raise KeyError(f"{self.name}: no uncertainty stage {method!r} "
                       f"(has {[u['method'] for u in self.uncertainty]})")

    def has(self, method: str) -> bool:
        return any(u["method"] == method for u in self.uncertainty)

    def inject_string(self) -> str:
        """The flat `name=value,name[i]=value` form parse_inject expects.

        Indexed knobs may be written either as a list (positional) or a mapping (sparse, when only some
        components move).  Both flatten to the same bracket form, and insertion order is preserved so the
        string is stable across loads.
        """
        out = []
        for k, v in self.data.inject.items():
            if isinstance(v, (list, tuple)):
                out += [f"{k}[{i}]={_num(x)}" for i, x in enumerate(v)]
            elif isinstance(v, dict):
                out += [f"{k}[{int(i)}]={_num(x)}" for i, x in v.items()]
            else:
                out.append(f"{k}={_num(v)}")
        return ",".join(out)

    def digest(self) -> str:
        """Stable hash of the resolved config -- goes into every stage manifest, so a merged product can
        prove its shards came from one definition."""
        return hashlib.sha256(yaml.safe_dump(self.as_dict(), sort_keys=True).encode()).hexdigest()[:12]

    def as_dict(self) -> dict:
        return {"name": self.name, "samples": list(self.samples), "beams": list(self.beams),
                "banks": vars(self.banks), "data": {"mode": self.data.mode,
                                                    "inject": self.data.inject,
                                                    "sigma": vars(self.data.sigma)},
                "fit": {"dials": self.fit.dials, "estimator": self.fit.estimator,
                        "minimizer": vars(self.fit.minimizer)},
                "uncertainty": [dict(u) for u in self.uncertainty]}


def _num(x) -> str:
    """Render a number for the flat inject-string form: no trailing zeros, no exponent."""
    s = f"{float(x):g}"
    return s
