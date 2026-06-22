"""Config dataclasses + YAML loader for the ADoNIS workflow API.

Two top-level configs, each loaded from a standardized YAML:
  - GenConfig       (generation: material, channels, cascade hyperparams, tracking, output)
  - AnalysisConfig  (analysis: input banks, signal topology, observables+binning, data overlay)

Number-precision sentinels: a cut/edge written as `cos70` resolves to float(np.cos(np.deg2rad(70.)))
and `pi` resolves to np.pi at load time, so YAML stays readable while cuts/bin edges remain
bit-identical to the hand-written scripts (cc1pi_engine_plot / cc0pi_engine_combined).
"""
from __future__ import annotations
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
import numpy as np
import yaml

COS70 = float(np.cos(np.deg2rad(70.0)))
_SCALAR_SENTINELS = {"cos70": COS70, "pi": float(np.pi), "-pi": -float(np.pi), "inf": float("inf")}


def _resolve_scalar(v):
    if isinstance(v, str) and v in _SCALAR_SENTINELS:
        return _SCALAR_SENTINELS[v]
    return v


def _resolve_seq(seq):
    return [_resolve_scalar(x) for x in seq]


def _coerce(cls, d):
    """Build dataclass `cls` from dict `d`, ignoring unknown keys is an ERROR (typo guard)."""
    if d is None:
        return cls()
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)} (allowed: {sorted(known)})")
    return cls(**d)


# ----------------------------------------------------------------------------- generation
@dataclass
class CascadeHyperparams:
    P: int = 12                 # BFS particle-buffer width
    max_gen: int = 6            # cascade generations
    n_recoil: int = 4           # top-K proton recoils per scatter (-> ADONIS_N_RECOIL)
    step: float = 0.04          # Glauber step [fm]
    max_steps: int = 260
    nn_inelastic: bool = True
    mprot: int = 6              # top-M proton terminals stored per event (-> ADONIS_MPROT)
    engine: str = "pool"        # cascade engine: "pool" (default, validated single core) | "bfs" (legacy)
                                # (single per-step-reconciled stack, ~2.8x faster, QE only so far)


@dataclass
class VegasConfig:
    enabled: bool = False       # RES importance estimator only: frozen VegasGrid over the 6 final-state
                                #   hypercube dims (beam + 3-body); default OFF -> bit-identical sampling
    nbins: int = 50             # per-axis grid bins
    warmup_iters: int = 6       # adapt iterations (accumulate -> refine), then freeze
    warmup_n: int = 100000      # events per warm-up iteration (per channel)
    alpha: float = 1.5          # VEGAS damping exponent (ACHILLES VegasParams::alpha_default)
    seed: int = 987654321       # warm-up RNG seed (fixed -> reproducible grid)
    cache: str = "auto"         # "auto" (load the sidecar grid if present, else build+save),
                                # "rebuild" (always warm up + overwrite the sidecar),
                                # "load" (require an existing sidecar; error if missing)
    grid_path: str | None = None  # explicit sidecar path; None -> <bank>_vegasgrid.npz next to the bank

    def __post_init__(self):
        if self.cache not in ("auto", "rebuild", "load"):
            raise ValueError(f"vegas.cache {self.cache!r} not in auto|rebuild|load")


@dataclass
class TrackingConfig:
    enabled: bool = False       # store per-event MC-truth summary (track/parent/pdg/end-process)
    steps: bool = False         # per-step trajectories (viz only; NOT in batch generation)
    max_tracks: int = 64
    max_steps: int = 260


@dataclass
class GenConfig:
    material: str = "C"
    n_per_seed: int = 30000
    n_seeds: int = 56
    channels: tuple = ("res",)              # subset of {"res","qe"}
    cascade: CascadeHyperparams = field(default_factory=CascadeHyperparams)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    vegas: VegasConfig = field(default_factory=VegasConfig)
    out_dir: str = "data/oracle"
    tag: str = ""
    seed0: int = 0
    fsi: bool = True            # False -> PRE-FSI bank (primary interaction products, no cascade)

    def __post_init__(self):
        bad = set(self.channels) - {"res", "qe"}
        if bad:
            raise ValueError(f"channels: {sorted(bad)} not in {{'res','qe'}}")


def load_gen_config(path) -> GenConfig:
    d = yaml.safe_load(Path(path).read_text()) or {}
    d.pop("name", None)
    d["cascade"] = _coerce(CascadeHyperparams, d.get("cascade"))
    d["tracking"] = _coerce(TrackingConfig, d.get("tracking"))
    d["vegas"] = _coerce(VegasConfig, d.get("vegas"))
    if "channels" in d:
        d["channels"] = tuple(d["channels"])
    return _coerce(GenConfig, d)


# ----------------------------------------------------------------------------- analysis
@dataclass
class SignalDef:
    """Signal topology, generalized over CC0pi & CC1pi (rich engine schema)."""
    mu_win: tuple = (250.0, 7000.0)
    p_win: tuple = (450.0, 1200.0)
    pi_win: tuple | None = (150.0, 1200.0)
    cos_mu: float | None = None             # CC0pi muon backward cut (e.g. -0.6); None -> use cth
    cth: float | None = COS70               # forward cos cut (cos70 CC1pi; 0.4 CC0pi proton)
    proton_lead: str = "in_window"          # "in_window" (CC1pi) | "global" (CC0pi NUISANCE def)
    proton_count: str = "ge1"               # "ge1" | "eq1"
    require_proton: bool = True
    pion_id: str = "pip"                    # "pip" | "anypi" | "none" (CC0pi: veto all pions)
    count_recoil_neutron: bool = False
    target: str = "carbon"                  # "carbon" | "hydrogen" | "CH"
    W_conv: str = "vertex"
    n_ejected: int | None = None            # require EXACTLY this many ejected protons (|p|>eject_thresh)
    eject_thresh: float = 250.0             # MeV; final-state proton momentum to count as "ejected"
    ref_proc: tuple | None = None           # restrict ACHILLES reference to these signal_process_id
                                            #   (200=QE, 401/402=RES); None = all modes

    def __post_init__(self):
        self.mu_win = tuple(_resolve_seq(self.mu_win))
        self.p_win = tuple(_resolve_seq(self.p_win))
        self.pi_win = None if self.pi_win is None else tuple(_resolve_seq(self.pi_win))
        self.cos_mu = _resolve_scalar(self.cos_mu)
        self.cth = _resolve_scalar(self.cth)
        if self.proton_lead not in ("in_window", "global"):
            raise ValueError(f"proton_lead {self.proton_lead!r} not in in_window|global")
        if self.pion_id not in ("pip", "anypi", "none"):
            raise ValueError(f"pion_id {self.pion_id!r} not in pip|anypi|none")
        if self.proton_count not in ("ge1", "eq0", "eq1", "eq2"):
            raise ValueError(f"proton_count {self.proton_count!r} not in ge1|eq0|eq1|eq2")
        if self.ref_proc is not None:
            self.ref_proc = tuple(int(p) for p in self.ref_proc)


@dataclass
class ObservableSpec:
    key: str
    label: str
    edges: list | None = None               # explicit edges (may contain sentinels e.g. "pi")
    linspace: list | None = None            # [lo, hi, n_edges] (sentinels allowed)

    def bin_edges(self) -> np.ndarray:
        if self.edges is not None:
            return np.asarray(_resolve_seq(self.edges), float)
        if self.linspace is not None:
            lo, hi, n = _resolve_seq(self.linspace)
            return np.linspace(float(lo), float(hi), int(n))
        raise ValueError(f"observable {self.key}: need edges or linspace")


@dataclass
class DataOverlay:
    enabled: bool = False
    source: str = "npz"                     # "npz" (t2k_cc0pi_stv_data.npz) | "nuisance_txt"
    path: str = ""
    names: dict = field(default_factory=dict)   # observable-key -> data key/file
    per_nucleon_cm2: bool = True
    A: int = 12


@dataclass
class AnalysisConfig:
    inputs: dict = field(default_factory=dict)      # roles: adonis_res, adonis_qe, adonis_h, reference
    signal: SignalDef = field(default_factory=SignalDef)
    observables: list = field(default_factory=list)
    data: DataOverlay = field(default_factory=DataOverlay)
    out_path: str = "paper_figures/adonis_analysis.png"
    title: str = ""
    ratio_band: tuple = (0.9, 1.1)
    ratio_ylim: tuple = (0.5, 1.6)
    carbon_only: bool = True


def load_analysis_config(path) -> AnalysisConfig:
    d = yaml.safe_load(Path(path).read_text()) or {}
    d.pop("name", None)
    d["signal"] = _coerce(SignalDef, d.get("signal"))
    d["data"] = _coerce(DataOverlay, d.get("data"))
    d["observables"] = [_coerce(ObservableSpec, o) for o in d.get("observables", [])]
    for k in ("ratio_band", "ratio_ylim"):
        if k in d:
            d[k] = tuple(d[k])
    return _coerce(AnalysisConfig, d)
