"""Identical-config replay: run ADoNIS's pion selection on ACHILLES's EXACT per-event nucleus configs.

Parses an ACHILLES_DUMPCFG + ACHILLES_VERTEXDUMP log (pi+ beam, CrossSection mode): per event the BEAM
pion + every background NUCleon (pid, 4-mom, position) BEFORE Evolve, and the VTX channel (ACHILLES's
outcome).  Then runs ADoNIS's `_pion_step` cascade on the SAME nuclei (same positions AND Fermi momenta),
so the candidate sets are bit-identical -- any difference in the proton-hit / CEX fraction is PURELY the
selection/transport algorithm.

Usage: python -u scripts/cex_replay.py output/achilles/_dc_pip.log
"""
import re
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adonis.workflow.materials import resolve_targets
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _load_density, _pion_step


def parse(logpath):
    """-> list of events: dict(beam=(p4,pos), nuc=[(pid,p4,pos)...], ach_channels=[...])."""
    events = []; cur = None
    for line in open(logpath):
        if line.startswith("DUMP_EVENT"):
            if cur is not None: events.append(cur)
            cur = dict(beam=None, nuc=[], chan=[], vtx=[])
        elif cur is None:
            continue
        elif line[:5] in ("BEAM ", "NUC p"):
            m = dict(re.findall(r"(\w+)=(-?[\d.]+(?:[eE][-+]?\d+)?)", line))
            if not all(k in m for k in ("pid", "px", "py", "pz", "E", "x", "y", "z")):
                cur["bad"] = True; continue          # interleaved/truncated line -> drop this event
            p4 = np.array([float(m["E"]), float(m["px"]), float(m["py"]), float(m["pz"])])
            pos = np.array([float(m["x"]), float(m["y"]), float(m["z"])])
            if line.startswith("BEAM"): cur["beam"] = (p4, pos)
            else: cur["nuc"].append((int(m["pid"]), p4, pos))
        elif line.startswith("VTX ") and "inc_pid=211" in line:
            mm = dict(re.findall(r"(\w+)=(-?[\d.]+(?:[eE][-+]?\d+)?)", line))
            if int(mm["channel"]) in (1, 2):
                cur["chan"].append(int(mm["channel"]))                 # first-vertex list (legacy)
                cur["vtx"].append((float(mm["inc_p"]), int(mm["channel"])))  # ALL pi+ vertices (inc_p, chan)
    if cur is not None: events.append(cur)
    return events


def main(logpath):
    events = [e for e in parse(logpath) if e["beam"] is not None and e["nuc"] and not e.get("bad")]
    A = max(len(e["nuc"]) for e in events)
    n = len(events)
    print(f"parsed {n} events, A(max)={A}", flush=True)
    # ACHILLES outcome on these configs (first vertex channel per event)
    ach_first = np.array([e["chan"][0] if e["chan"] else 0 for e in events])  # 0 none,1 el,2 cex
    # build batched su (pad to A with dead nucleons far away)
    npos = np.zeros((n, A, 3)); nmom = np.tile([939.0, 0, 0, 0], (n, A, 1)).astype(float)
    nisp = np.zeros((n, A), bool); nmask = np.zeros((n, A), bool)
    beam_p4 = np.zeros((n, 4)); beam_pos = np.zeros((n, 3))
    for i, e in enumerate(events):
        beam_p4[i] = e["beam"][0]; beam_pos[i] = e["beam"][1]
        for j, (pid, p4, pos) in enumerate(e["nuc"]):
            npos[i, j] = pos; nmom[i, j] = p4; nisp[i, j] = (pid == 2212); nmask[i, j] = True
    npos = jnp.asarray(npos); nmom = jnp.asarray(nmom); nisp = jnp.asarray(nisp)
    consumed = jnp.asarray(~nmask)                       # padded slots pre-consumed (never candidates)
    tg = resolve_targets("C")[0][0]
    cfg = DiscreteCascadeConfig(nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs,
                                step=0.04, pauli=True, nn_inelastic=True)
    rg, rp, rn, radius = _load_density(cfg.nucleus, cfg.density_n)
    p4 = jnp.asarray(beam_p4); pos = jnp.asarray(beam_pos)
    d3 = p4[:, 1:]; dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
    ch = jnp.zeros(n, jnp.int32); nsc = jnp.zeros(n, jnp.int32); alive = jnp.ones(n, bool)
    key = jax.random.PRNGKey(0)
    step = jax.jit(_pion_step, static_argnums=(14,))
    # ALL pi+ scatters on the same starting configs: record (plab, cex?) for vertices where pion is STILL pi+
    a_plab, a_cex = [], []
    for st in range(1200):
        kP = jax.random.split(jax.random.fold_in(key, st), n)
        chp = np.asarray(ch); al = np.asarray(alive)
        plab_pre = np.asarray(jnp.linalg.norm(p4[:, 1:], axis=1))
        (p4n, posn, dhn, chn, nscn, aln), esc, is_abs, is_conv, s1, s2, consn, _ = step(
            p4, pos, dhat, ch, nsc, alive, npos, nmom, nisp, consumed, rg, rp, rn, radius, cfg, kP)
        scat = (np.asarray(nscn) > np.asarray(nsc)) & al & (chp == 0)   # vertex where pion is pi+
        if scat.any():
            idx = np.where(scat)[0]
            a_plab.append(plab_pre[idx]); a_cex.append((np.asarray(chn)[idx] != chp[idx]).astype(np.int32))
        p4, pos, dhat, ch, nsc, consumed = p4n, posn, dhn, chn, nscn, consn
        alive = aln & ~jnp.asarray(np.asarray(is_abs) | np.asarray(is_conv))   # scatters CONTINUE
        if bool(jnp.all(~alive)): break
    ap = np.concatenate(a_plab); ak = np.concatenate(a_cex)
    # ACHILLES: ALL pi+ vertices (inc_p, chan) on the SAME configs
    cv = np.array([v for e in events for v in e["vtx"]]); cp_, cc = cv[:, 0], cv[:, 1].astype(int)
    print(f"\nALL pi+ SCATTERS on IDENTICAL configs ({n} events) -- CEX fraction per |p| (charge-matched):")
    print(f"  {'bin':12}{'ADoNIS':>9}{'ACHILLES':>10}{'ratio':>8}")
    for lo, hi in [(200, 400), (400, 600), (600, 900)]:
        a = (ap >= lo) & (ap < hi); c = (cp_ >= lo) & (cp_ < hi)
        fa = ak[a].mean(); fc = (cc[c] == 2).mean(); ea = np.sqrt(fa*(1-fa)/a.sum()); ec = np.sqrt(fc*(1-fc)/c.sum())
        r = fa/fc; er = r*np.sqrt((ea/fa)**2+(ec/fc)**2)
        print(f"  [{lo},{hi})    {fa:9.3f}{fc:10.3f}{r:7.3f}±{er:.3f}   (nA={int(a.sum())} nC={int(c.sum())})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/achilles/_dc_pip.log")
