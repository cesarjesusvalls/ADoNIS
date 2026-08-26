"""Config dataclasses + YAML loader for the ADoNIS workflow API.

Two top-level configs, each loaded from a standardized YAML:
  - GenConfig       (generation: material, channels, cascade hyperparams, tracking, output)
  - AnalysisConfig  (analysis: input banks, signal topology, observables+binning, data overlay)

Number-precision sentinels: a cut/edge written as `cos70` resolves to float(np.cos(np.deg2rad(70.)))
and `pi` resolves to np.pi at load time, so YAML stays readable while cuts/bin edges remain
exact.
"""
from __future__ import annotations
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
import numpy as np
import yaml

from adonis.constants import COS70
_SCALAR_SENTINELS = {"cos70": COS70, "pi": float(np.pi), "-pi": -float(np.pi), "inf": float("inf")}


def _resolve_scalar(v):
    if isinstance(v, str) and v in _SCALAR_SENTINELS:
        return _SCALAR_SENTINELS[v]
    return v


def _resolve_seq(seq):
    return [_resolve_scalar(x) for x in seq]


def _coerce(cls, d):
    """Build dataclass `cls` from dict `d`. Unknown keys raise (typo guard), not ignored."""
    if d is None:
        return cls()
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)} (allowed: {sorted(known)})")
    return cls(**d)


@dataclass
class CascadeHyperparams:
    step: float = 0.04
    max_steps: int = 260
    nn_inelastic: bool = True
    time_step: bool = False
    path_budget_R: float = 20.0
    mprot: int = 6


@dataclass
class VegasConfig:
    enabled: bool = False
    nbins: int = 50
    warmup_iters: int = 6
    warmup_n: int = 100000
    alpha: float = 1.5
    seed: int = 987654321
    cache: str = "auto"
    grid_path: str | None = None

    def __post_init__(self):
        if self.cache not in ("auto", "rebuild", "load"):
            raise ValueError(f"vegas.cache {self.cache!r} not in auto|rebuild|load")


@dataclass
class TrackingConfig:
    enabled: bool = False
    steps: bool = False
    max_tracks: int = 64
    max_steps: int = 260


FLUX_FILES = {
    "t2k":        "flux/T2K_nu.dat",
    "minerva":    "flux/minerva_numu_fhc.dat",
    "microboone": "flux/microboone_numu.dat",
}


PROBES = ("CC", "NC", "EM", "hadron")
HADRON_BEAMS = ("pip", "prot", "neut")
GEN_BEAMS = ("spectrum", "electron") + HADRON_BEAMS
from adonis.flux.electron import E_BEAM_JLAB


_PROBE_BEAMS = {"CC": ("spectrum",), "NC": ("spectrum",), "EM": ("electron",),
                "hadron": HADRON_BEAMS}


@dataclass
class GenConfig:
    """Generation config -- one schema for every bank, used by adonis.workflow.generate_bank.
    `probe` selects the primary interaction:
      * CC     : charged-current neutrino hard vertex (channels qe/res), beam=spectrum (a flux table)
      * NC     : neutral-current neutrino hard vertex, beam=spectrum.  theta_acc MUST be full
                 acceptance -- a polar cut on an invisible outgoing neutrino is meaningless and would
                 silently bias the sample, so it is rejected rather than ignored.
      * EM     : electron hard vertex (channels qe/res), beam=electron (monochromatic e-), theta_acc cut
      * hadron : a tagged pi+/p/n projectile (no hard vertex, pure FSI transport), beam in {pip,prot,neut},
                 |p| uniform in [pmin,pmax]
    `fsi` (default True) runs the cascade -> the rich reweight records; fsi=False -> a pre-FSI bank."""
    probe: str = "CC"
    beam: str = "spectrum"
    material: str = "C"
    channels: tuple = ("res",)
    flux: str = "t2k"
    e_beam: float = E_BEAM_JLAB
    theta_acc: tuple = (0.0, 180.0)
    pmin: float = 50.0
    pmax: float = 1000.0
    achilles_coupl1_quirk: bool = False
    fsi: bool = True
    pauli: bool = True
    cascade: CascadeHyperparams = field(default_factory=CascadeHyperparams)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    vegas: VegasConfig = field(default_factory=VegasConfig)
    n_w: int | None = None
    n_per_seed: int = 30000
    n_seeds: int = 56
    seed0: int = 0
    chunk: int | None = None
    out_dir: str = "output/adonis"
    tag: str = ""

    def __post_init__(self):
        if self.probe not in PROBES:
            raise ValueError(f"probe {self.probe!r} not in {sorted(PROBES)}")
        if self.beam not in GEN_BEAMS:
            raise ValueError(f"beam {self.beam!r} not in {sorted(GEN_BEAMS)}")
        if self.beam not in _PROBE_BEAMS[self.probe]:
            raise ValueError(f"probe={self.probe!r} requires beam in {_PROBE_BEAMS[self.probe]}, got {self.beam!r}")
        if self.probe in ("CC", "NC", "EM"):
            bad = set(self.channels) - {"res", "qe"}
            if bad:
                raise ValueError(f"channels: {sorted(bad)} not in {{'res','qe'}} (probe {self.probe})")
        self.theta_acc = tuple(_resolve_seq(self.theta_acc))
        if self.probe == "NC" and not (self.theta_acc[0] <= 0.0 and self.theta_acc[1] >= 180.0):
            raise ValueError(f"probe='NC' requires full theta_acc (0,180), got {self.theta_acc}: the "
                             "outgoing lepton is a neutrino, so an angular acceptance on it is "
                             "meaningless and would silently bias the sample")
        if self.probe != "NC" and self.achilles_coupl1_quirk:
            raise ValueError("achilles_coupl1_quirk is an NC QE coupling switch; it has no meaning "
                             f"for probe={self.probe!r} and setting it there would be a silent no-op")
        if self.probe == "hadron" and self.pmax <= self.pmin:
            raise ValueError(f"pmax {self.pmax} must exceed pmin {self.pmin}")
        if self.beam == "spectrum":
            if self.flux not in FLUX_FILES:
                raise ValueError(f"flux {self.flux!r} not in {sorted(FLUX_FILES)}")
            import os
            want = FLUX_FILES[self.flux]; have = os.environ.get("ADONIS_FLUX_FILE")
            if have is not None and have != want:
                raise ValueError(f"flux key {self.flux!r} expects ADONIS_FLUX_FILE={want!r} but env has {have!r}")

    @property
    def bank_prefix(self) -> str:
        """Bank-name beam tag: flux key (CC), ee<E> (EM), or the projectile name (hadron)."""
        if self.probe == "hadron":
            return self.beam
        return self.flux if self.beam == "spectrum" else f"ee{int(round(self.e_beam))}"


def load_gen_config(path) -> GenConfig:
    d = yaml.safe_load(Path(path).read_text()) or {}
    d.pop("name", None)
    d["cascade"] = _coerce(CascadeHyperparams, d.get("cascade"))
    d["tracking"] = _coerce(TrackingConfig, d.get("tracking"))
    d["vegas"] = _coerce(VegasConfig, d.get("vegas"))
    if "channels" in d:
        d["channels"] = tuple(d["channels"])
    if "theta_acc" in d:
        d["theta_acc"] = tuple(d["theta_acc"])
    return _coerce(GenConfig, d)




@dataclass
class NuSignalDef:
    """Neutrino signal topology, covering CC0pi, CC1pi, and NC1pi0.  Sibling EleBeamSignalDef below
    covers the electron-beam (e,e') figures; the top-level `probe:` key (default "nu") selects which
    one a config coerces into."""
    mu_win: tuple = (250.0, 7000.0)
    p_win: tuple = (450.0, 1200.0)
    pi_win: tuple | None = (150.0, 1200.0)
    cos_mu: float | None = None
    cth: float | None = None
    proton_lead: str = "in_window"
    proton_count: str = "ge1"
    require_proton: bool = True
    pt_hi: float | None = None
    pz_win: tuple | None = None
    pion_id: str = "pip"
    count_recoil_neutron: bool = False
    target: str = "carbon"
    W_conv: str = "vertex"
    tpi_win: tuple | None = None
    w_exp_max: float | None = None
    veto_other_mesons: bool = False
    n_ejected: int | None = None
    eject_thresh: float = 250.0
    ref_proc: tuple | None = None

    def __post_init__(self):
        self.mu_win = tuple(_resolve_seq(self.mu_win))
        self.p_win = tuple(_resolve_seq(self.p_win))
        self.pi_win = None if self.pi_win is None else tuple(_resolve_seq(self.pi_win))
        self.tpi_win = None if self.tpi_win is None else tuple(_resolve_seq(self.tpi_win))
        self.pz_win = None if self.pz_win is None else tuple(_resolve_seq(self.pz_win))
        self.cos_mu = _resolve_scalar(self.cos_mu)
        self.cth = _resolve_scalar(self.cth)
        if self.proton_lead not in ("in_window", "global"):
            raise ValueError(f"proton_lead {self.proton_lead!r} not in in_window|global")
        if self.pion_id not in ("pip", "pi0", "anypi", "none"):
            raise ValueError(f"pion_id {self.pion_id!r} not in pip|pi0|anypi|none")
        if self.proton_count not in ("ge1", "eq0", "eq1", "eq2"):
            raise ValueError(f"proton_count {self.proton_count!r} not in ge1|eq0|eq1|eq2")
        if self.ref_proc is not None:
            self.ref_proc = tuple(int(p) for p in self.ref_proc)


@dataclass
class EleBeamSignalDef:
    """Electron-beam (e,e') acceptance + reconstruction constants -- the sibling of NuSignalDef for the
    electron-scattering figures (inclusive (e,e') omega, and e4nu E_QE/E_cal/P_T).  Its reducers are
    selection.ele_signal / ele_oracle_signal.  Fields default to "no cut" so the inclusive figure (fig01,
    which needs only the electron polar window) leaves the proton/removal fields unset."""
    beam_energy: float = 0.0
    e_min: float | None = None
    e_theta_win: tuple = (0.0, 180.0)
    omega_win: tuple | None = None
    p_min: float | None = None
    p_theta_win: tuple | None = None
    removal_energy: float = 0.0

    def __post_init__(self):
        self.e_theta_win = tuple(_resolve_seq(self.e_theta_win))
        if self.p_theta_win is not None:
            self.p_theta_win = tuple(_resolve_seq(self.p_theta_win))
        if self.omega_win is not None:
            self.omega_win = tuple(_resolve_seq(self.omega_win))


@dataclass
class ObservableSpec:
    key: str
    label: str
    edges: list | None = None
    linspace: list | None = None
    fit: bool = True

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
    source: str = "npz"
    path: str = ""
    names: dict = field(default_factory=dict)
    per_nucleon_cm2: bool = True
    A: int = 12


@dataclass
class AnalysisConfig:
    inputs: dict = field(default_factory=dict)
    probe: str = "nu"
    signal: object = field(default_factory=NuSignalDef)
    observables: list = field(default_factory=list)
    data: DataOverlay = field(default_factory=DataOverlay)
    out_path: str = "output/analysis.png"
    title: str = ""
    ratio_band: tuple = (0.9, 1.1)
    ratio_ylim: tuple = (0.5, 1.6)
    carbon_only: bool = True
    legend_loc: str = ""


def load_analysis_config(path) -> AnalysisConfig:
    d = yaml.safe_load(Path(path).read_text()) or {}
    for _k in ("name", "render", "compute", "params", "layout", "heavy"):
        d.pop(_k, None)
    _SEL = {"nu": NuSignalDef, "electron": EleBeamSignalDef}
    probe = d.get("probe", "nu")
    if probe not in _SEL:
        raise ValueError(f"probe {probe!r} not in {sorted(_SEL)}")
    d["signal"] = _coerce(_SEL[probe], d.get("signal"))
    d["data"] = _coerce(DataOverlay, d.get("data"))
    d["observables"] = [_coerce(ObservableSpec, o) for o in d.get("observables", [])]
    for k in ("ratio_band", "ratio_ylim"):
        if k in d:
            d[k] = tuple(d[k])
    return _coerce(AnalysisConfig, d)


SignalDef = NuSignalDef
