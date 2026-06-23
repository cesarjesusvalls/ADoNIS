"""ADoNIS CASCADE-SEGMENT generator (reusable, material-driven) -- the ADoNIS half of the FULL cascade
matrix (all interactions, primaries AND secondaries), produced by the IN-ENGINE jitted logger that lives
inside the real pool (run_cascade_pool log_cap path) -- NOT a re-implementation.  Every cascade SEGMENT
(a particle's free-flight episode ending in an interaction or a terminal escape) is recorded with full
G4-like provenance.  Schema (per segment):
  proc(0=QE/1=RES), inc_pid, inc_p[MeV], channel, nprod, prod_pid[3], prod_p[3], w,
  is_first(1 = the FIRST segment of an original primary -> filter to recover the primary matrix),
  gen, track_id, parent_id
Channels: pion {0 transmit,1 elastic,2 charge-ex,3 absorption,4 conversion}; nucleon {0 transmit,
1 elastic NN->NN, 2 inelastic NN->NNpi}.  ACHILLES emits the SAME schema (achilles:vertex, all-segment
mode); cascade_vertex_matrix.py compares them (use --primaries to filter is_first==1).

Run: python -u scripts/gen_cascade_segments.py [C|Ar] [n_per_seed] [seeds] [out.npz] [max_steps] [P] [Lcap]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAT = sys.argv[1] if len(sys.argv) > 1 else "C"
NPS = int(sys.argv[2]) if len(sys.argv) > 2 else 80000
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
OUT = sys.argv[4] if len(sys.argv) > 4 else f"/tmp/cascade_segments_{MAT}_ado.npz"
MAXSTEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
P_BUF = int(sys.argv[6]) if len(sys.argv) > 6 else 16
LCAP = int(sys.argv[7]) if len(sys.argv) > 7 else 64
os.environ.setdefault("ADONIS_N_RECOIL", "8")

import numpy as np, jax, jax.numpy as jnp
np.seterr(all="ignore")
from adonis.workflow.materials import resolve_targets
from adonis.xsec import res_xsec, qe_xsec
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec.spectral import SpectralFunction
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _load_density
from adonis.fsi.cascade_full import cascade_nucleus
CHUNK = 12000


def build_cfg(tg, ms):
    return DiscreteCascadeConfig(step=0.04, max_steps=ms, seed=1, nn_inelastic=True, pauli=True,
                                 nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)


def run_logged(channel, p_pi, p_N, Npid, ipid, cfg, key, pid_pi=None):
    """Run a chunk through THE production cascade (cascade_nucleus) with the in-engine logger on -- the
    SAME entry point the cross-section forward generation uses, just log_cap=L.  No duplicated cascade.
    Returns the (n,L) log dict + per-event segment counts + overflow."""
    n = p_N.shape[0]
    if channel == "qe":                                              # QE proton: no primary pion, struck n
        p_pi = jnp.zeros((n, 4)); pid_pi = jnp.zeros(n, jnp.int32); pid_Ni = jnp.full(n, 2112, jnp.int32)
    else:                                                            # RES: primary pion + recoil nucleon
        pid_pi = jnp.asarray(pid_pi, jnp.int32); pid_Ni = jnp.asarray(ipid, jnp.int32)
    log, counts, logofl = cascade_nucleus(p_pi, p_N, pid_pi, pid_Ni, jnp.asarray(Npid, jnp.int32),
                                         cfg, key, P=P_BUF, channel=channel, log_cap=LCAP)
    return {k: np.asarray(v) for k, v in log.items()}, np.asarray(counts), int(logofl)


def convert(log, counts, proc, w):
    """Flatten the (n,L) log to per-segment records + assemble products + compute is_first."""
    n, L = log["chan"].shape
    valid = np.arange(L)[None, :] < counts[:, None]
    gen = log["gen"]; tid = log["track_id"]
    # is_first: the FIRST (earliest-col == chronological) gen-0 segment per (event, track_id).
    g0v = (gen == 0) & valid
    is_first = np.zeros((n, L), bool)
    for c in range(L):
        seen = ((tid[:, :c] == tid[:, c:c + 1]) & g0v[:, :c]).any(axis=1) if c else np.zeros(n, bool)
        is_first[:, c] = g0v[:, c] & ~seen
    # products: continuing particle (channels 1,2) + alive daughters n1,n2,pio
    chan = log["chan"]
    cont_in = (chan == 1) | (chan == 2)
    pp_pid = np.zeros((n, L, 3), np.int64); pp_p = np.zeros((n, L, 3), np.float64)
    nprod = np.zeros((n, L), np.int64)
    pieces = [(cont_in, log["cont_pid"], log["cont_p"]),
              (log["n1_al"].astype(bool), log["n1_pid"], log["n1_p"]),
              (log["n2_al"].astype(bool), log["n2_pid"], log["n2_p"]),
              (log["pio_al"].astype(bool), log["pio_pid"], log["pio_p"])]
    for keep, pid, pmom in pieces:
        slot = np.clip(nprod, 0, 2)
        ii, jj = np.where(keep & valid)
        s = nprod[ii, jj]
        ok = s < 3
        pp_pid[ii[ok], jj[ok], s[ok]] = pid[ii[ok], jj[ok]]
        pp_p[ii[ok], jj[ok], s[ok]] = pmom[ii[ok], jj[ok]]
        nprod = nprod + (keep & valid).astype(np.int64)
    nprod = np.clip(nprod, 0, 3)
    m = valid
    wseg = np.broadcast_to(w[:, None], (n, L))
    return dict(inc_pid=log["inc_pid"][m], inc_p=log["incp"][m], channel=chan[m], nprod=nprod[m],
                prod_pid=pp_pid[m], prod_p=pp_p[m], w=wseg[m],
                proc=np.full(int(m.sum()), proc, np.int64), is_first=is_first[m].astype(np.int64),
                gen=gen[m], track_id=tid[m], parent_id=log["parent_id"][m])


def main():
    tg = resolve_targets(MAT)[0][0]
    _, _, _, radius = _load_density(tg.density_p, tg.density_n)
    ms = max(MAXSTEPS, int(np.ceil(3.0 * radius / 0.04)))
    cfg = build_cfg(tg, ms)
    sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)
    print(f"[cfg] {MAT} radius={radius:.2f} max_steps={ms} P={P_BUF} Lcap={LCAP} SEEDS={SEEDS} n/seed={NPS}", flush=True)
    acc = {k: [] for k in ("inc_pid", "inc_p", "channel", "nprod", "prod_pid", "prod_p", "w", "proc",
                           "is_first", "gen", "track_id", "parent_id")}
    ofl_tot = [0]

    def add(D):
        for k in acc:
            acc[k].append(D[k])

    def checkpoint(ns):
        Dall = {k: np.concatenate(acc[k]) for k in acc}
        Dall["w"] = Dall["w"] / ns
        np.savez(OUT, material=MAT, seeds_done=ns, log_overflow=ofl_tot[0], **Dall)
        print(f"  [checkpoint] wrote {OUT} after {ns}/{SEEDS} seed(s)  (segments={len(Dall['w'])}, logofl={ofl_tot[0]})", flush=True)

    for sd in range(SEEDS):
        e = res_xsec.generate(NPS, seed=sd, return_events=True, sf_n=sf_n, sf_p=sf_p,
                              n_neutron=tg.A - tg.Z, n_proton=tg.Z)["events"]
        rw = np.asarray(e["w"]); rs = rw > 0
        ppi = np.asarray(e["p_pi"])[rs]; ppid = np.asarray(e["ppid"])[rs]
        pN = np.asarray(e["p_N"])[rs]; Npid = np.asarray(e["Npid"])[rs]; ipid = np.asarray(e["ipid"])[rs]; rw = rw[rs]
        q = qe_xsec.sample_importance(NPS, seed=sd, sf=sf_n, n_neutron=tg.A - tg.Z)
        qN = np.asarray(q["p_out"]); qw = np.asarray(q["w"]) / NPS; qNpid = np.full(len(qw), 2212)
        nrc = (len(rw) + CHUNK - 1) // CHUNK
        for ci, c0 in enumerate(range(0, len(rw), CHUNK)):
            sl = slice(c0, c0 + CHUNK)
            log, counts, ofl = run_logged("res", jnp.asarray(ppi[sl]), jnp.asarray(pN[sl]),
                                          jnp.asarray(Npid[sl], jnp.int32), ipid[sl], cfg,
                                          jax.random.PRNGKey(31 + sd), pid_pi=ppid[sl])
            add(convert(log, counts, 1, rw[sl])); ofl_tot[0] += ofl
            print(f"    seed {sd+1}/{SEEDS} RES chunk {ci+1}/{nrc} (+{len(rw[sl])} events, logofl+{ofl})", flush=True)
        for ci, c0 in enumerate(range(0, len(qw), CHUNK)):
            sl = slice(c0, c0 + CHUNK)
            log, counts, ofl = run_logged("qe", None, jnp.asarray(qN[sl]),
                                          jnp.asarray(qNpid[sl], jnp.int32), None, cfg, jax.random.PRNGKey(33 + sd))
            add(convert(log, counts, 0, qw[sl])); ofl_tot[0] += ofl
            print(f"    seed {sd+1}/{SEEDS} QE  chunk {ci+1}/{(len(qw)+CHUNK-1)//CHUNK} (+{len(qw[sl])} events, logofl+{ofl})", flush=True)
        print(f"  seed {sd+1}/{SEEDS} done", flush=True)
        checkpoint(sd + 1)
    print(f"\nwrote {OUT}  (final, {SEEDS} seeds, log_overflow={ofl_tot[0]})", flush=True)


if __name__ == "__main__":
    main()
