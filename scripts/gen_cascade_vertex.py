"""ADoNIS CASCADE-VERTEX generator (reusable, material-driven) -- the ADoNIS half of the
cascade-vertex matrix (analog of the cross-section matrix gen_cc_matrix).

For each PRIMARY cascade particle (RES pi + RES recoil nucleon + QE proton), run its FIRST interaction
through the REAL per-step physics (_pion_step / _nucleon_step) and emit ONE record:
  inc_pid, inc_p [MeV], channel, n_prod, prod_pid[K], prod_p[K], w
Transmitted primaries (escaped before interacting) are recorded with channel=0 so interaction
FRACTIONS have a denominator.  Channels (self-documenting codes):
  pion:    0 transmit | 1 elastic (scatter, same charge) | 2 charge-exchange | 3 absorption | 4 conversion
  nucleon: 0 transmit | 1 elastic (NN->NN) | 2 inelastic (NN->NDelta->NN pi)
Products carry pid + lab |p| (MeV).  Material via resolve_targets (single source of truth: density/
configs/spectral).  ACHILLES emits the SAME schema (ACHILLES_VERTEXDUMP); cascade_vertex_matrix.py
compares them per incident-type x channel x product-type, vs incident |p| and product |p|.

Run: python -u scripts/gen_cascade_vertex.py [C|Ar] [n_per_seed] [seeds] [out.npz] [max_steps]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAT = sys.argv[1] if len(sys.argv) > 1 else "C"
NPS = int(sys.argv[2]) if len(sys.argv) > 2 else 80000
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
OUT = sys.argv[4] if len(sys.argv) > 4 else f"/tmp/cascade_vertex_{MAT}_ado.npz"
MAXSTEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
os.environ.setdefault("ADONIS_N_RECOIL", "8")

import numpy as np, jax, jax.numpy as jnp
np.seterr(all="ignore")
from adonis.workflow.materials import resolve_targets
from adonis.xsec import res_xsec, qe_xsec
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec.spectral import SpectralFunction
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
from adonis.fsi.cascade_discrete import (DiscreteCascadeConfig, _load_density, _pion_step, _nucleon_step)
from adonis.fsi.cascade_full import setup_nucleus
from adonis.fsi.cascade_real import _CH_PID                       # charge idx (0:+,1:0,2:-) -> pion pid

CHUNK = 20000
_PIDPI = np.array([211, 111, -211])                              # charge idx 0/1/2 -> pid


def build_cfg(tg, ms):
    return DiscreteCascadeConfig(step=0.04, max_steps=ms, seed=1, nn_inelastic=True, pauli=True,
                                 nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)


def _emptyrec(n):
    return dict(inc_pid=np.zeros(n, np.int64), inc_p=np.zeros(n), channel=np.zeros(n, np.int64),
                nprod=np.zeros(n, np.int64), prod_pid=np.zeros((n, 3), np.int64),
                prod_p=np.zeros((n, 3), np.float64), done=np.zeros(n, bool))


def _set_prod(R, idx, plist):
    """plist: list of (pid, p4) for the products at this vertex (np arrays over the active mask idx)."""
    R["nprod"][idx] = len(plist)
    for k, (pid, p) in enumerate(plist):
        R["prod_pid"][idx, k] = pid
        R["prod_p"][idx, k] = p


def run_pions(p_pi, ppid, ipid, cfg, su, rgrid, rhoP, rhoN, radius, key):
    """Single-pass each primary pion to its FIRST interaction; classify channel + products.

    Channel keys off the ACTUAL post-Pauli-blocking outcomes the engine propagates: is_abs / is_conv
    (returned) and a scatter is signalled by the scatter-counter `nsc` incrementing this step.  A
    Pauli-blocked candidate (geometric hit but blocked) is NOT a vertex -- the pion keeps propagating
    and may interact later or transmit, exactly as ACHILLES does.  (Keying off the pre-block `bcode`
    would over-count abs/elastic.)"""
    n = p_pi.shape[0]
    R = _emptyrec(n); R["inc_pid"] = np.asarray(ppid); R["inc_p"] = np.linalg.norm(np.asarray(p_pi)[:, 1:], axis=1)
    ch_init = np.asarray(su["ch0"]).astype(np.int64)
    p4 = p_pi; pos = su["pos0"]; ch = su["ch0"]; nsc = jnp.zeros(n, jnp.int32); alive = jnp.ones(n, bool)
    consumed = su["consumed0"]; keys = jax.random.split(key, MAXSTEPS)
    for i in range(MAXSTEPS):
        nsc_prev = np.asarray(nsc)
        d3 = p4[:, 1:]; dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
        (p4, pos, _dp, ch, nsc, alive), esc, is_abs, is_conv, s1, s2, consumed, srec = _pion_step(
            p4, pos, dhat, ch, nsc, alive, su["npos"], su["nmom"], su["nisp"], consumed,
            rgrid, rhoP, rhoN, radius, cfg, keys[i])
        chn = np.asarray(ch).astype(np.int64)
        is_scat = (np.asarray(nsc) > nsc_prev) & ~R["done"]       # ACTUAL scatter this step
        new_abs = np.asarray(is_abs) & ~R["done"]                 # ACTUAL absorption (post-block)
        new_conv = np.asarray(is_conv) & ~R["done"]               # ACTUAL conversion
        s1p4 = np.asarray(s1[0]); s1q = np.asarray(s1[3]); s2p4 = np.asarray(s2[0]); s2q = np.asarray(s2[3])
        pip = np.linalg.norm(np.asarray(p4)[:, 1:], axis=1)
        s1p = np.linalg.norm(s1p4[:, 1:], axis=1); s2p = np.linalg.norm(s2p4[:, 1:], axis=1)
        # ELASTIC vs CHARGE-EX: out pion charge (chn) vs incident charge (ch_init)
        for mask, code in ((is_scat & (chn == ch_init), 1), (is_scat & (chn != ch_init), 2)):
            ix = np.where(mask)[0]
            for j in ix:                                          # out pion + recoil nucleon
                R["channel"][j] = code; R["nprod"][j] = 2
                R["prod_pid"][j, 0] = _PIDPI[chn[j]]; R["prod_p"][j, 0] = pip[j]
                R["prod_pid"][j, 1] = 2212 if s1q[j] == 1 else 2112; R["prod_p"][j, 1] = s1p[j]
            R["done"][ix] = True
        ix = np.where(new_abs)[0]                                 # absorption: 2 nucleons
        for j in ix:
            R["channel"][j] = 3; R["nprod"][j] = 2
            R["prod_pid"][j, 0] = 2212 if s1q[j] == 1 else 2112; R["prod_p"][j, 0] = s1p[j]
            R["prod_pid"][j, 1] = 2212 if s2q[j] == 1 else 2112; R["prod_p"][j, 1] = s2p[j]
        R["done"][ix] = True
        ix = np.where(new_conv)[0]                                # conversion: baryon
        for j in ix:
            R["channel"][j] = 4; R["nprod"][j] = 1
            R["prod_pid"][j, 0] = 2212 if s1q[j] == 1 else 2112; R["prod_p"][j, 0] = s1p[j]
        R["done"][ix] = True
        if R["done"].all() or not bool(jnp.any(alive)):
            break
    return R


def run_nucleons(p_N, Npid, ipid, cfg, su, rgrid, rhoP, rhoN, radius, key):
    """Single-pass each primary nucleon to its FIRST interaction; elastic (NN->NN) vs inelastic (NN->NDelta->Npi)."""
    n = p_N.shape[0]
    R = _emptyrec(n); R["inc_pid"] = np.asarray(Npid); R["inc_p"] = np.linalg.norm(np.asarray(p_N)[:, 1:], axis=1)
    isp = jnp.asarray(np.asarray(Npid) == 2212)
    p4 = p_N; pos = su["pos0"]; fz = jnp.zeros(n); alive = jnp.ones(n, bool); consumed = su["consumed0"]
    keys = jax.random.split(key, MAXSTEPS)
    for i in range(MAXSTEPS):
        d3 = p4[:, 1:]; dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
        (p4, pos, _dn, fz, alive), esc, recap, do, ko, pin, consumed, nstat = _nucleon_step(
            p4, pos, dhat, fz, isp, alive, su["npos"], su["nmom"], su["nisp"], consumed,
            rgrid, rhoP, rhoN, radius, cfg, keys[i])
        do = np.asarray(do).astype(bool)                         # ACTUAL elastic scatter (post-block)
        madepi = np.asarray(pin[4])                              # ACTUAL inelastic (pi_alive, post-block)
        ko_p4 = np.asarray(ko[0]); ko_q = np.asarray(ko[3]); ko_al = np.asarray(ko[4])
        pi4 = np.asarray(pin[0]); pich = np.asarray(pin[3]).astype(np.int64)
        nucp = np.linalg.norm(np.asarray(p4)[:, 1:], axis=1)
        kop = np.linalg.norm(ko_p4[:, 1:], axis=1); pip = np.linalg.norm(pi4[:, 1:], axis=1)
        new_inel = madepi & ~R["done"]
        new_el = do & ~madepi & ~R["done"]
        ix = np.where(new_inel)[0]                               # NN -> N N pi
        for j in ix:
            # NOTE: the pool keeps the primary nucleon's charge through an inelastic step
            # (cascade_full.py chg_2 = where(is_pi, chp, chg)) -> the leading is propagated as `isp`.
            # We record what the engine actually propagates (the matrix will surface any p/n mismatch
            # vs ACHILLES, which assigns the leading its Delta-decay charge).  FLAGGED in the audit.
            R["channel"][j] = 2; R["nprod"][j] = 3
            R["prod_pid"][j, 0] = 2212 if isp[j] else 2112; R["prod_p"][j, 0] = nucp[j]   # scattered primary
            R["prod_pid"][j, 1] = 2212 if ko_q[j] == 1 else 2112; R["prod_p"][j, 1] = kop[j]  # 2nd nucleon
            R["prod_pid"][j, 2] = _PIDPI[pich[j]]; R["prod_p"][j, 2] = pip[j]              # created pion
        R["done"][ix] = True
        ix = np.where(new_el)[0]                                 # NN -> N N (elastic)
        for j in ix:
            R["channel"][j] = 1; R["nprod"][j] = 2
            R["prod_pid"][j, 0] = 2212 if isp[j] else 2112; R["prod_p"][j, 0] = nucp[j]
            R["prod_pid"][j, 1] = 2212 if ko_q[j] == 1 else 2112; R["prod_p"][j, 1] = kop[j]
        R["done"][ix] = True
        if R["done"].all() or not bool(jnp.any(alive)):
            break
    return R


def main():
    tg = resolve_targets(MAT)[0][0]
    _, _, _, radius = _load_density(tg.density_p, tg.density_n)
    ms = max(MAXSTEPS, int(np.ceil(3.0 * radius / 0.04)))
    cfg = build_cfg(tg, ms)
    rgrid, rhoP, rhoN, rad = _load_density(tg.density_p, tg.density_n)
    sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)
    print(f"[cfg] {MAT} radius={rad:.2f} max_steps={ms} SEEDS={SEEDS} n/seed={NPS}", flush=True)
    acc = {k: [] for k in ("inc_pid", "inc_p", "channel", "nprod", "prod_pid", "prod_p", "w", "proc")}

    def add(R, w, proc):                                          # proc: 0=QE, 1=RES (mirror the xsec matrix split)
        for k in ("inc_pid", "inc_p", "channel", "nprod", "prod_pid", "prod_p"):
            acc[k].append(R[k])
        acc["w"].append(w); acc["proc"].append(np.full(len(w), proc, np.int64))

    def checkpoint(nseeds_done):                                  # incremental write -> early matrix comparisons
        D = {k: np.concatenate(acc[k]) for k in acc}
        D["w"] = D["w"] / nseeds_done                             # average over the seeds INCLUDED so far
        np.savez(OUT, material=MAT, seeds_done=nseeds_done, **D)
        print(f"  [checkpoint] wrote {OUT} after {nseeds_done}/{SEEDS} seed(s)  (N={len(D['w'])})", flush=True)
        return D

    for sd in range(SEEDS):
        e = res_xsec.generate(NPS, seed=sd, return_events=True, sf_n=sf_n, sf_p=sf_p,
                              n_neutron=tg.A - tg.Z, n_proton=tg.Z)["events"]
        rw = np.asarray(e["w"]); rs = rw > 0
        ppi = jnp.asarray(np.asarray(e["p_pi"])[rs]); ppid = np.asarray(e["ppid"])[rs]
        pN = jnp.asarray(np.asarray(e["p_N"])[rs]); Npid = np.asarray(e["Npid"])[rs]; ipid = np.asarray(e["ipid"])[rs]
        rw = rw[rs]
        # QE primary proton
        q = qe_xsec.sample_importance(NPS, seed=sd, sf=sf_n, n_neutron=tg.A - tg.Z)
        qN = jnp.asarray(q["p_out"]); qw = np.asarray(q["w"]) / NPS; qNpid = np.full(len(qw), 2212)
        nrc = (len(rw) + CHUNK - 1) // CHUNK
        for ci, c0 in enumerate(range(0, len(rw), CHUNK)):
            sl = slice(c0, c0 + CHUNK)
            su = setup_nucleus(ppi[sl], jnp.asarray(ppid[sl], jnp.int32), jnp.asarray(ipid[sl], jnp.int32), cfg, jax.random.PRNGKey(11 + sd))
            add(run_pions(ppi[sl], ppid[sl], ipid[sl], cfg, su, rgrid, rhoP, rhoN, rad, jax.random.PRNGKey(31 + sd)), rw[sl], 1)
            suN = setup_nucleus(pN[sl], jnp.zeros(len(Npid[sl]), jnp.int32), jnp.asarray(ipid[sl], jnp.int32), cfg, jax.random.PRNGKey(12 + sd))
            add(run_nucleons(pN[sl], Npid[sl], ipid[sl], cfg, suN, rgrid, rhoP, rhoN, rad, jax.random.PRNGKey(32 + sd)), rw[sl], 1)
            print(f"    seed {sd+1}/{SEEDS} RES chunk {ci+1}/{nrc} (+{len(rw[sl])} pi +{len(rw[sl])} N)", flush=True)
        for ci, c0 in enumerate(range(0, len(qw), CHUNK)):
            sl = slice(c0, c0 + CHUNK)
            suQ = setup_nucleus(qN[sl], jnp.zeros(len(qNpid[sl]), jnp.int32), jnp.full(len(qNpid[sl]), 2112, jnp.int32), cfg, jax.random.PRNGKey(13 + sd))
            add(run_nucleons(qN[sl], qNpid[sl], np.full(len(qNpid[sl]), 2112), cfg, suQ, rgrid, rhoP, rhoN, rad, jax.random.PRNGKey(33 + sd)), qw[sl], 0)
            print(f"    seed {sd+1}/{SEEDS} QE  chunk {ci+1}/{(len(qw)+CHUNK-1)//CHUNK} (+{len(qw[sl])} N)", flush=True)
        print(f"  seed {sd+1}/{SEEDS} done", flush=True)
        checkpoint(sd + 1)                                        # write after EVERY seed for early comparison
    D = {k: np.concatenate(acc[k]) for k in acc}; D["w"] = D["w"] / SEEDS
    # quick console summary: interaction-channel fractions per incident type
    _summary(D)
    print(f"\nwrote {OUT}  (final, {SEEDS} seeds)", flush=True)


def _summary(D):
    PNAME = {211: "pi+", 111: "pi0", -211: "pi-", 2212: "p", 2112: "n"}
    CH = {0: "transmit", 1: "elastic", 2: "charge-ex/inel", 3: "absorption", 4: "conversion"}
    PROC = {0: "QE", 1: "RES"}
    for proc in (0, 1):                                           # split by production process (xsec-matrix style)
        pm = D["proc"] == proc
        if not pm.any(): continue
        print(f"\n=== ADoNIS {PROC[proc]} primary first-interaction channel fractions (weighted) ===")
        for pid in (211, 111, -211, 2212, 2112):
            m = pm & (D["inc_pid"] == pid); w = D["w"]
            if not m.any() or w[m].sum() <= 0: continue
            wm = w[m]; line = f"{PNAME[pid]:>4s} (N={int(m.sum())}): "
            for c in (0, 1, 2, 3, 4):
                f = float((wm * (D["channel"][m] == c)).sum() / wm.sum())
                if f > 1e-4: line += f"{CH[c]}={f:.3f}  "
            print(line)


if __name__ == "__main__":
    main()
