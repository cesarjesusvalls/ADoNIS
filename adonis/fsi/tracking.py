"""Geant4-style MC-truth tracking for the differentiable cascade -- MODULAR and channel-agnostic.

The cascade (adonis/fsi/cascade_full) is a BFS of Markov walk-segments through a shared nucleus.  This
module records, per event, a fixed-shape "truth" of every particle that was tracked:

  TrackBank (summary, cheap -- toggle `track`):  per track (n, MAX_TRACKS)
    track_id   : unique id within the event (gen*GEN_STRIDE + flat slot)
    parent_id  : track_id of the spawning particle  (-1 for gen-0 primaries)
    pdg        : PDG code (211/111/-211 pion, 2212/2112 nucleon, 0 absorbed, -1 converted)
    end_process: PROC_* enum (how the track ended: escape / absorb / convert / truncated / none)
    gen, origin: BFS depth ; gen-0 ancestor chain (0 = RES/QE nucleon, 1 = pion-knockout)
    p4_birth/p4_death (4) ; pos_birth (3)
    daughters  : derived ONCE at finalize() from parent_id (numpy, outside JIT)

  StepTrace (trajectory, viz -- toggle `track_steps`):  (n, MAX_TRACKS, MAX_STEPS, 7) float32
    per step: position (3) + momentum-4 (4).  Written from the march scan; differentiable w.r.t.
    inputs/theta -> demonstrates genuine sampling in a fully-differentiable system.

Both toggles are independent and DEFAULT OFF: with track=False the engine path is bit-exact production
(no allocation, no cost).  Memory per event (float32 steps): summary ~MAX_TRACKS*0.15 KB; full trajectory
~MAX_TRACKS*MAX_STEPS*28 B (steps are viz-only -> small N).  See docs/logbook/cascade_tracker.md.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# ---- end-process enum (supersets cascade_full FATE_NONE/ESCAPE/ABSORB/CONVERT = 0/1/2/3) ----
PROC_NONE, PROC_ESCAPE, PROC_ABSORB, PROC_CONVERT, PROC_TRUNCATED = 0, 1, 2, 3, 4
PROC_NAMES = {0: "none", 1: "escape", 2: "absorb", 3: "convert", 4: "truncated"}

GEN_STRIDE = 1000          # track_id = GEN_STRIDE*gen + flat_slot  (flat_slot < P*K << GEN_STRIDE)
_NULL = -1                 # parent_id of a gen-0 primary; track_id of an empty slot
STEP_C = 7                 # per-step channels: pos(3) + p4(4)


@dataclass
class TrackerConfig:
    """Toggle the tracker.  Both default OFF -> production path untouched / bit-exact."""
    track: bool = False            # record the per-track summary (TrackBank)
    track_steps: bool = False      # also record the per-step trajectory (StepTrace) -- viz, small N
    max_tracks: int = 64           # fixed # of track slots per event (overflow logged)
    max_steps: int = 260           # fixed trajectory length (= cfg.max_steps)

    @property
    def on(self):
        return self.track or self.track_steps


# ---------- summary assembly (from the engine's per-generation terminal batches) ----------
_SUM_FIELDS = ("track_id", "parent_id", "pdg", "end_process", "gen", "origin")


def assemble_summary(terminals):
    """terminals: list of per-generation terminal batch dicts, each (n, P_g) carrying
    track_id, parent_id, pid, fate, gen, origin, p4, pos(optional), alive.  Concatenate along the
    track axis -> per-track summary arrays (n, sum_g P_g).  `alive` marks slots that were real tracks."""
    cat = lambda k: np.concatenate([np.asarray(t[k]) for t in terminals], axis=1)
    out = dict(track_id=cat("track_id"), parent_id=cat("parent_id"), pdg=cat("pid"),
               end_process=cat("fate"), gen=cat("gen"), origin=cat("origin"),
               p4_death=cat("p4"), real=cat("alive"))
    if "pos" in terminals[0]:
        out["pos_birth"] = cat("pos")
    if "p4_birth" in terminals[0]:
        out["p4_birth"] = cat("p4_birth")
    return out


def derive_daughters(summary, max_daughters=8):
    """Build, per event, the immediate-daughter index lists from parent_id (numpy, outside JIT).
    Returns daughters (n, MAX_TRACKS, max_daughters) of child SLOT indices (-1 padded)."""
    tid = summary["track_id"]; pid = summary["parent_id"]; real = summary["real"]
    n, T = tid.shape
    dau = np.full((n, T, max_daughters), _NULL, np.int64)
    for e in range(n):
        # map this event's track_id -> slot
        id2slot = {int(tid[e, s]): s for s in range(T) if real[e, s]}
        cnt = np.zeros(T, np.int64)
        for s in range(T):
            if not real[e, s]:
                continue
            par = int(pid[e, s])
            ps = id2slot.get(par, _NULL)
            if ps != _NULL and cnt[ps] < max_daughters:
                dau[e, ps, cnt[ps]] = s
                cnt[ps] += 1
    return dau


def finalize(terminals, step_trace=None, cfg: TrackerConfig = None, max_daughters=8):
    """Pack the engine outputs into a clean MC-truth dict:
      summary fields + 'daughters' (n, T, max_daughters) + optional 'steps' (n, T, MAX_STEPS, 7)."""
    s = assemble_summary(terminals)
    s["daughters"] = derive_daughters(s, max_daughters)
    s["n_tracks"] = s["real"].sum(1)                       # per-event real track count
    if step_trace is not None:
        s["steps"] = np.asarray(step_trace)                # (n, T, MAX_STEPS, 7) float32
    return s


# ---------- memory accounting (the explicit estimate) ----------
def memory_per_event(cfg: TrackerConfig, max_daughters=8):
    """Bytes per event for the chosen toggles (float32 steps)."""
    T = cfg.max_tracks
    summary = T * (6 * 4 + 4 * 8 + 4 * 8 + 3 * 8) + T * max_daughters * 8     # ids/enums int32 + p4s/pos f64
    steps = T * cfg.max_steps * STEP_C * 4 if cfg.track_steps else 0          # float32
    return dict(summary_B=summary, steps_B=steps, total_B=summary + steps)


# ---------- per-step trajectory diagnostics (from march track_steps) ----------
def termination_step(traj_pos, radius):
    """Physical termination = first step at which |pos| exceeds the nuclear radius (the particle has
    left the nucleus).  traj_pos (n_steps, n, 3) -> per-particle step index (n_steps if it never exits)."""
    outside = np.linalg.norm(np.asarray(traj_pos), axis=2) > radius        # (n_steps, n)
    ns = outside.shape[0]
    return np.where(outside.any(0), outside.argmax(0), ns)


def step_summary(traj_pos, radius, max_steps):
    """Median / 90% / 99% termination step + fraction still inside at the cap -- the 'how long does the
    march need' diagnostic, computed from a real run's trajectories (no separate sweep)."""
    ts = termination_step(traj_pos, radius)
    return dict(median=float(np.median(ts)), p90=float(np.percentile(ts, 90)),
                p99=float(np.percentile(ts, 99)), max=int(ts.max()),
                frac_inside_at_cap=float(np.mean(ts >= max_steps)))


# ---------- text dump of one event's tree (diagnostic) ----------
def print_tree(truth, e=0):
    """Pretty-print event e's track tree (track_id [pdg] gen origin -> end_process)."""
    tid = truth["track_id"][e]; par = truth["parent_id"][e]; pdg = truth["pdg"][e]
    ep = truth["end_process"][e]; gen = truth["gen"][e]; real = truth["real"][e]
    print(f"event {e}: {int(real.sum())} tracks")
    for s in np.argsort(gen, kind="stable"):
        if not real[s]:
            continue
        print(f"  id={int(tid[s]):5d} pdg={int(pdg[s]):5d} gen={int(gen[s])} "
              f"parent={int(par[s]):5d} -> {PROC_NAMES.get(int(ep[s]), '?')}")
