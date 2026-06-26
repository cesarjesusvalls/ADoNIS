"""Parse the ACHILLES CASCADEDUMP stderr stream (ACHILLES_CASCADEDUMP=1) into per-event arrays for the
hybrid ACHILLES<->ADoNIS cascade ablation harness (scripts/cascade_ablation.py).

Per event the dump emits (see Achilles src/Achilles/Cascade.cc INSTR CASCADEDUMP):
  CDEVT  id w proc                                              -- event header
  CDPRIM id pid E px py pz x y z                                -- each propagating primary (QE proton)
  CDBG   id idx pid E px py pz x y z                            -- each background nucleon (struck removed)
  CDSCAT id struck_idx type blocked pid1 pid2 in1=.. in2=.. nout out=.. out=..   -- each NN interaction
  CDFS   id pid E px py pz                                      -- each final-state nucleon/pion (escaped)
  CDEND  id

Returns a list of event dicts: {id, w, proc, prim:[(pid,p4,pos)], bg:[(idx,pid,p4,pos)],
scat:[{struck_idx,type,blocked,pid1,pid2,in1,in2,out:[(pid,p4)]}], fs:[(pid,p4)]}.
"""
import numpy as np


def _kv(tokens):
    """parse 'k=v' tokens (v may be 'a,b,c,d') into a dict of strings."""
    d = {}
    for t in tokens:
        if "=" in t:
            k, v = t.split("=", 1)
            d[k] = v
    return d


def _vec(s):
    return np.array([float(x) for x in s.split(",")], dtype=np.float64)


def parse_cascadedump(path):
    events = {}
    order = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            if not line[:2] == "CD":
                continue
            tok = line.split()
            tag = tok[0]
            d = _kv(tok[1:])
            eid = int(d["id"])
            if tag == "CDEVT":
                events[eid] = dict(id=eid, w=float(d["w"]), proc=int(d["proc"]),
                                   prim=[], bg=[], scat=[], fs=[])
                order.append(eid)
            elif tag == "CDPRIM":
                e = events[eid]
                e["prim"].append((int(d["pid"]),
                                  np.array([float(d["E"]), float(d["px"]), float(d["py"]), float(d["pz"])]),
                                  np.array([float(d["x"]), float(d["y"]), float(d["z"])])))
            elif tag == "CDBG":
                e = events[eid]
                e["bg"].append((int(d["idx"]), int(d["pid"]),
                                np.array([float(d["E"]), float(d["px"]), float(d["py"]), float(d["pz"])]),
                                np.array([float(d["x"]), float(d["y"]), float(d["z"])])))
            elif tag == "CDSCAT":
                e = events[eid]
                # 'out=' appears multiple times -> _kv keeps only the last; re-scan tokens for all out=
                outs = []
                for t in tok[1:]:
                    if t.startswith("out="):
                        parts = t[4:].split(",")
                        outs.append((int(parts[0]), np.array([float(x) for x in parts[1:]])))
                e["scat"].append(dict(struck_idx=int(d["struck_idx"]), type=int(d["type"]),
                                      blocked=int(d["blocked"]), pid1=int(d["pid1"]), pid2=int(d["pid2"]),
                                      in1=_vec(d["in1"]), in2=_vec(d["in2"]), out=outs))
            elif tag == "CDFS":
                e = events[eid]
                e["fs"].append((int(d["pid"]),
                                np.array([float(d["E"]), float(d["px"]), float(d["py"]), float(d["pz"])])))
            elif tag == "CDEND":
                pass
    return [events[i] for i in order if i in events]


def to_arrays(events, A_pad=None):
    """Pack parsed events into batched arrays for ADoNIS su_external ingestion.
    Returns dict with prim_p4 (n,4), prim_pos (n,3), prim_pid (n,), npos (n,A,3), nmom (n,A,4),
    nisp (n,A) proton-mask, nmask (n,A) valid-slot mask, and ACHILLES final-state counts ach_np/ach_nn (n,)
    plus the per-event ejected nucleon momenta lists ach_fs (list)."""
    n = len(events)
    A = A_pad or max(len(e["bg"]) for e in events)
    prim_p4 = np.zeros((n, 4)); prim_pos = np.zeros((n, 3)); prim_pid = np.zeros(n, np.int64)
    npos = np.zeros((n, A, 3)); nmom = np.zeros((n, A, 4)); nmom[:, :, 0] = 939.0
    nisp = np.zeros((n, A), bool); nmask = np.zeros((n, A), bool)
    ach_np = np.zeros(n, np.int64); ach_nn = np.zeros(n, np.int64); w = np.zeros(n)
    for i, e in enumerate(events):
        # the (highest-energy) primary nucleon
        prims = [p for p in e["prim"] if abs(p[0]) in (2212, 2112)]
        if prims:
            p0 = max(prims, key=lambda p: p[1][0])
            prim_pid[i], prim_p4[i], prim_pos[i] = p0[0], p0[1], p0[2]
        for a, (idx, pid, p4, pos) in enumerate(e["bg"][:A]):
            npos[i, a] = pos; nmom[i, a] = p4; nisp[i, a] = (pid == 2212); nmask[i, a] = True
        for pid, p4 in e["fs"]:
            if pid == 2212: ach_np[i] += 1
            elif pid == 2112: ach_nn[i] += 1
        w[i] = e["w"]
    return dict(prim_p4=prim_p4, prim_pos=prim_pos, prim_pid=prim_pid, npos=npos, nmom=nmom,
                nisp=nisp, nmask=nmask, ach_np=ach_np, ach_nn=ach_nn, w=w, A=A)


if __name__ == "__main__":
    import sys
    evs = parse_cascadedump(sys.argv[1])
    print(f"parsed {len(evs)} events")
    if evs:
        e = evs[0]
        print(f"  evt0: prim={len(e['prim'])} bg={len(e['bg'])} scat={len(e['scat'])} fs={len(e['fs'])}")
        arr = to_arrays(evs)
        print(f"  A(bg)={arr['A']}  mean bg protons={arr['nisp'].sum(1).mean():.2f}  "
              f"mean ACH N(p)={arr['ach_np'].mean():.3f}  N(n)={arr['ach_nn'].mean():.3f}")
