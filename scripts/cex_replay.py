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
            cur = dict(beam=None, nuc=[], chan=[])
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
            if int(mm["channel"]) in (1, 2): cur["chan"].append(int(mm["channel"]))
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
    ado_kind = np.full(n, -1, np.int32); ado_rq = np.full(n, -9, np.int32); done = np.zeros(n, bool)
    step = jax.jit(_pion_step, static_argnums=(14,))
    for st in range(1200):
        kP = jax.random.split(jax.random.fold_in(key, st), n)
        chp = np.asarray(ch)
        (p4n, posn, dhn, chn, nscn, aln), esc, is_abs, is_conv, s1, s2, consn, _ = step(
            p4, pos, dhat, ch, nsc, alive, npos, nmom, nisp, consumed, rg, rp, rn, radius, cfg, kP)
        scat = (np.asarray(nscn) > np.asarray(nsc)) & ~done
        rem = (np.asarray(is_abs) | np.asarray(is_conv)) & ~done
        if scat.any():
            idx = np.where(scat)[0]
            ado_kind[idx] = (np.asarray(chn)[idx] != chp[idx]).astype(np.int32)   # 0 el, 1 cex
            ado_rq[idx] = np.asarray(s1[3])[idx]; done[idx] = True
        if rem.any():
            idx = np.where(rem)[0]; ado_kind[idx] = 2; done[idx] = True            # abs/conv
        p4, pos, dhat, ch, nsc, consumed = p4n, posn, dhn, chn, nscn, consn
        alive = aln & ~jnp.asarray(done)
        if bool(jnp.all(~alive)): break
    # compare on SCATTERS (el+cex), BINNED by beam |p| -- the axis where the deficit appeared
    plab = np.linalg.norm(beam_p4[:, 1:], axis=1)
    np.savez("/tmp/cex_replay.npz", plab=plab, ado_kind=ado_kind, ado_rq=ado_rq, ach_first=ach_first)
    am = (ado_kind == 0) | (ado_kind == 1); cm = (ach_first == 1) | (ach_first == 2)
    print(f"\nON IDENTICAL ACHILLES CONFIGS ({n} events) -- CEX fraction per beam |p|:")
    print(f"  {'bin':12}{'ADoNIS':>9}{'ACHILLES':>10}{'ratio':>8}")
    for lo, hi in [(80, 200), (200, 400), (400, 600), (600, 900)]:
        a = am & (plab >= lo) & (plab < hi); c = cm & (plab >= lo) & (plab < hi)
        if a.sum() > 30 and c.sum() > 30:
            fa = (ado_kind[a] == 1).mean(); fc = (ach_first[c] == 2).mean()
            print(f"  [{lo},{hi})    {fa:9.3f}{fc:10.3f}{fa/fc:8.3f}   (nA={int(a.sum())} nC={int(c.sum())})")
    both = am & cm
    print(f"  same-channel (el/cex) where BOTH scatter ({int(both.sum())}): "
          f"{(ado_kind[both]==(ach_first[both]-1)).mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/achilles/_dc_pip.log")
