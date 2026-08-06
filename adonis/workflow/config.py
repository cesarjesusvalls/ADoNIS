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

from adonis.constants import COS70   # single source (adonis.constants)
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
    # NOTE: the pool stack width is FIXED at 1 (serial, ACHILLES-faithful processing order) -- there is no
    # P knob.  See adonis/fsi/cascade_full.py (cascade_nucleus hardcodes M=1).
    step: float = 0.04          # Glauber step [fm] (time_step=False: distance/step; True: Dt/step)
    max_steps: int = 260        # legacy field; the engine now uses a fixed 100k runaway ceiling +
                                #   path_budget_R*radius as the physics bound (generate.py sets the ceiling)
    nn_inelastic: bool = True
    time_step: bool = False     # stepping clock: False = distance-sync (every particle sweeps `step`);
                                #   True = ACHILLES AdaptiveStep time-sync (Dt=step/beta_max per round)
    path_budget_R: float = 20.0  # runaway backstop (was 3.0): drop a particle once its path length > this
                                #   * radius.  20R never clips a real escaping/capturing track (net escape
                                #   ~1R); at 3R it silently clipped ~0.009% slow random-walkers -> reacted
                                #   mis-count.  Keep in sync with DiscreteCascadeConfig + tune.POOLCFG.
    mprot: int = 6              # top-M proton terminals stored per event (-> ADONIS_MPROT)


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


# flux key -> ACHILLES Spectrum table (relative to the sibling Achilles/ dir).  The generators read
# the actual table via adonis.flux.spectrum (ADONIS_FLUX_FILE) (env ADONIS_FLUX_FILE); a run must export that to
# the file below so the bank name (this key) and the physics agree.  See _resolve_flux() in the CLI.
FLUX_FILES = {
    "t2k":        "flux/T2K_nu.dat",
    "minerva":    "flux/minerva_numu_fhc.dat",
    "microboone": "flux/microboone_numu.dat",
}


# PROBE NAMES ARE THE PHYSICS, and they must stay that way: adonis.channels.probes is the registry
# that decides what each one MEANS, and "weak" was a name that could not survive NC -- a neutral
# current is every bit as weak as a charged one, so "weak" would have had to mean "CC except when it
# means NC".  Renamed to "CC" outright; there is deliberately NO alias and no back-compat mapping,
# because a mapping is how the lie survives.  load_gen_config rejects unknown keys and probe_spec()
# raises on unknown values, so a missed rename fails loudly rather than silently meaning CC.
PROBES = ("CC", "NC", "EM", "hadron")   # CC/NC = charged-/neutral-current neutrino ; EM = electron ;
#                                     hadron = a tagged hadron projectile (no hard vertex, pure FSI transport)
HADRON_BEAMS = ("pip", "prot", "neut")               # tagged-hadron projectiles (adonis.flux.hadron.BEAMS)
GEN_BEAMS = ("spectrum", "electron") + HADRON_BEAMS  # nu spectrum | mono e- | pi+/p/n projectile
E_BEAM_JLAB = 2222.0            # default monochromatic e- energy [MeV] (JLab 2.222 GeV); adonis.flux.electron


# probe -> the beam sources it is allowed to pair with (a bank can't be mislabelled across probes).
_PROBE_BEAMS = {"CC": ("spectrum",), "NC": ("spectrum",), "EM": ("electron",),
                "hadron": HADRON_BEAMS}


@dataclass
class GenConfig:
    """THE single generation config -- one schema for every bank the pipeline makes, driven entirely by
    fields (there is exactly one generator, adonis.workflow.generate_bank; the diversity is here, not in
    the code path).  `probe` selects the primary interaction:
      * CC     : charged-current neutrino hard vertex (channels qe/res), beam=spectrum (a flux table)
      * NC     : neutral-current neutrino hard vertex, beam=spectrum.  theta_acc MUST be full
                 acceptance -- a polar cut on an invisible outgoing neutrino is meaningless and would
                 silently bias the sample, so it is rejected rather than ignored.
      * EM     : electron hard vertex (channels qe/res), beam=electron (monochromatic e-), theta_acc cut
      * hadron : a tagged pi+/p/n projectile (no hard vertex, pure FSI transport), beam in {pip,prot,neut},
                 |p| uniform in [pmin,pmax]
    `fsi` (default True) runs the cascade -> the rich reweight records; fsi=False -> a pre-FSI bank."""
    probe: str = "CC"           # CC | NC | EM | hadron
    beam: str = "spectrum"      # spectrum | electron | pip | prot | neut  (must be consistent with probe)
    material: str = "C"
    channels: tuple = ("res",)              # CC/NC/EM: subset of {"res","qe"}; ignored for hadron
    # --- CC (neutrino) ---
    flux: str = "t2k"           # neutrino flux key (beam=spectrum only); see FLUX_FILES
    # --- EM (electron) ---
    e_beam: float = E_BEAM_JLAB             # monochromatic e- energy [MeV]
    theta_acc: tuple = (0.0, 180.0)         # outgoing-lepton polar acceptance [deg], applied UNIFORMLY to
    #                                         every hard-vertex channel (CC muon + EM electron) at
    #                                         generation.  Default (0,180) = full acceptance (no-op);
    #                                         EM configs set (5,180) to cut the forward 1/q^4 divergence.
    # --- hadron (tagged beam) ---
    pmin: float = 50.0                      # projectile |p| window [MeV/c] (uniform)
    pmax: float = 1000.0
    # --- cascade / FSI ---
    # --- NC ---
    achilles_coupl1_quirk: bool = False     # NC QE only.  False = correct physics (the SM coupling);
    #                                         True = reproduce ACHILLES's coupl1 sin2w/sw discrepancy
    #                                         verbatim (~1.0396 on both nucleons' F1/F2).  Set True in
    #                                         the bank config that feeds the ACHILLES-comparison
    #                                         figures, so the comparison is like-for-like.  RECORDED IN
    #                                         THE MANIFEST -- a bank can never be ambiguous about which
    #                                         convention produced it.  See channels/currents/dirac.py.
    fsi: bool = True                        # False -> PRE-FSI bank (primary products, no cascade)
    pauli: bool = True                      # cascade Pauli blocking (False -> DEBUG ablation)
    cascade: CascadeHyperparams = field(default_factory=CascadeHyperparams)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    vegas: VegasConfig = field(default_factory=VegasConfig)
    n_w: int | None = None                  # cascade refill working set (None=engine default; 0=lock-step)
    # --- sharding / output ---
    n_per_seed: int = 30000
    n_seeds: int = 56
    seed0: int = 0
    chunk: int | None = None                # events/chunk (dense FSI buffers scale w/ this); None -> n_per_seed
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


# Tagged-beam generation is no longer a separate config/driver: a hadron beam is just GenConfig with
# probe="hadron", beam in HADRON_BEAMS -- built by the ONE generate_bank(cfg).  (BeamGenConfig + its
# load_beam_config retired in P3; adonis.flux.hadron.BEAMS is the single projectile registry.)


# ----------------------------------------------------------------------------- analysis
@dataclass
class NuSignalDef:
    """Neutrino signal topology, generalized over CC0pi & CC1pi (and NC1pi0).  The canonical selection
    schema; sibling EleBeamSignalDef below covers the electron-beam (e,e') figures.  Pick which one a
    config coerces into via the top-level `probe:` key (default "nu")."""
    mu_win: tuple = (250.0, 7000.0)
    p_win: tuple = (450.0, 1200.0)
    pi_win: tuple | None = (150.0, 1200.0)
    cos_mu: float | None = None             # CC0pi muon backward cut (e.g. -0.6); None -> use cth
    cth: float | None = COS70               # forward cos cut (cos70 CC1pi; 0.4 CC0pi proton)
    proton_lead: str = "in_window"          # "in_window" (CC1pi) | "global" (CC0pi NUISANCE def)
    proton_count: str = "ge1"               # "ge1" | "eq1"
    require_proton: bool = True
    pt_hi: float | None = None              # muon transverse-momentum cap [MeV] (MINERvA qelike pT/pz box)
    pz_win: tuple | None = None             # muon longitudinal-momentum window [MeV] (MINERvA qelike);
    #                                         with require_proton=False these give the hadron-inclusive
    #                                         isCC0pi_MINERvAPTPZ phase space (0 mesons, theta_mu<20).
    pion_id: str = "pip"                    # "pip" | "pi0" | "anypi" | "none" (CC0pi: veto all pions)
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
    beam_energy: float = 0.0            # MeV; incident electron energy E_beam
    e_min: float | None = None          # MeV; scattered-electron energy floor E_e (None -> no cut)
    e_theta_win: tuple = (0.0, 180.0)   # deg; scattered-electron polar acceptance [lo, hi]
    p_min: float | None = None          # MeV; proton momentum floor for the 1p0pi topology (None -> no lead)
    p_theta_win: tuple | None = None    # deg; proton polar acceptance [lo, hi] (None -> no lead)
    removal_energy: float = 0.0         # MeV; nuclear removal/binding energy epsilon in E_QE / E_cal

    def __post_init__(self):
        self.e_theta_win = tuple(_resolve_seq(self.e_theta_win))
        if self.p_theta_win is not None:
            self.p_theta_win = tuple(_resolve_seq(self.p_theta_win))


@dataclass
class ObservableSpec:
    key: str
    label: str
    edges: list | None = None               # explicit edges (may contain sentinels e.g. "pi")
    linspace: list | None = None            # [lo, hi, n_edges] (sentinels allowed)
    fit: bool = True                        # enters the Gate-I Fisher / fit?  False = plot-only validation
    #                                         extra (e.g. ppi/cos_pi, MINERvA dphit/lp_p) NUISANCE never
    #                                         released -- sec1 still plots it, the gradient ignores it.

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
    source: str = "npz"                     # "npz" (data/experiment/t2k_cc0pi_stv/) | "nuisance_txt"
    path: str = ""
    names: dict = field(default_factory=dict)   # observable-key -> data key/file
    per_nucleon_cm2: bool = True
    A: int = 12


@dataclass
class AnalysisConfig:
    inputs: dict = field(default_factory=dict)      # roles: adonis_res, adonis_qe, adonis_h, reference
    probe: str = "nu"                               # "nu" -> signal is NuSignalDef; "electron" -> EleBeamSignalDef
    signal: object = field(default_factory=NuSignalDef)
    observables: list = field(default_factory=list)
    data: DataOverlay = field(default_factory=DataOverlay)
    out_path: str = "paper_figures/adonis_analysis.png"
    title: str = ""
    ratio_band: tuple = (0.9, 1.1)
    ratio_ylim: tuple = (0.5, 1.6)
    carbon_only: bool = True
    legend_loc: str = ""            # style hint: matplotlib loc for the panel legend ("" -> caller default)


def load_analysis_config(path) -> AnalysisConfig:
    d = yaml.safe_load(Path(path).read_text()) or {}
    # figure-orchestration keys consumed by analysis/paper/sec1_validation (make.py + helper.py): the
    # render hook, its compute/params, the make_figure layout, the --light flag.  They are NOT
    # AnalysisConfig fields, so pop them here -- a figure spec that ALSO carries a selection (fig10/fig11,
    # the electron figs) loads as an AnalysisConfig through the same door as the pure-selection multiobs specs.
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


# Deprecated back-compat alias: the class was renamed SignalDef -> NuSignalDef when the electron sibling
# (EleBeamSignalDef) was added.  External jobs/ scripts live outside this repo and may still import
# SignalDef by name; keep this alias until they are updated, then delete it.
SignalDef = NuSignalDef
