"""STANDALONE _KSLAB validation (Cat-3): does evaluating only the K=3 nearest in-slab nucleons per step
(cfg.fast_xsec=True, production default) bias the cascade vs evaluating ALL nucleons (fast_xsec=False,
the EXACT ground truth)?

A/B: run the SAME events (same key) through the logged pool with fast_xsec True vs False, and compare the
first-interaction channel fractions per (proc, incident type) -- the cascade-vertex-matrix metric.  No
engine edit (just the cfg flag).  Ar (A=40, the most in-slab candidates) is the worst case.

Run: python -u scripts/test_kslab_ab.py [C|Ar] [N]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAT = sys.argv[1] if len(sys.argv) > 1 else "Ar"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 24000
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
LCAP = 96


def cfg_for(tg, fast):
    _, _, _, radius = _load_density(tg.density_p, tg.density_n)
    ms = max(1000, int(np.ceil(3.0 * radius / 0.04)))
    return DiscreteCascadeConfig(step=0.04, max_steps=100000, seed=1, nn_inelastic=True, pauli=True,
                                 nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs,
                                 fast_xsec=fast)


def first_chan_fracs(log, counts, w):
    """first-segment (gen==0, first per track_id) channel fractions per incident pid (proc-agnostic here)."""
    n, L = log["chan"].shape
    valid = np.arange(L)[None, :] < np.asarray(counts)[:, None]
    gen = np.asarray(log["gen"]); tid = np.asarray(log["track_id"]); chan = np.asarray(log["chan"])
    ipid = np.asarray(log["inc_pid"])
    g0 = (gen == 0) & valid
    isf = np.zeros((n, L), bool)
    for c in range(L):
        seen = ((tid[:, :c] == tid[:, c:c + 1]) & g0[:, :c]).any(axis=1) if c else np.zeros(n, bool)
        isf[:, c] = g0[:, c] & ~seen
    m = isf
    ip = ipid[m]; ch = chan[m]; ww = np.repeat(w, m.sum(axis=1))
    out = {}
    for pid in (211, 111, -211, 2212, 2112):
        sel = ip == pid
        if not sel.any() or ww[sel].sum() <= 0:
            continue
        ws = ww[sel]; out[pid] = {c: float((ws * (ch[sel] == c)).sum() / ws.sum()) for c in range(5)}
    return out


def run(tg, fast, key):
    cfg = cfg_for(tg, fast)
    e = res_xsec.generate(N, seed=0, return_events=True, sf_n=SpectralFunction(tg.spectral_n),
                          sf_p=SpectralFunction(tg.spectral_p), n_neutron=tg.A - tg.Z, n_proton=tg.Z)["events"]
    rw = np.asarray(e["w"]); rs = rw > 0
    ppi = jnp.asarray(np.asarray(e["p_pi"])[rs]); ppid = jnp.asarray(np.asarray(e["ppid"])[rs], jnp.int32)
    pN = jnp.asarray(np.asarray(e["p_N"])[rs]); Npid = jnp.asarray(np.asarray(e["Npid"])[rs], jnp.int32)
    ipid = jnp.asarray(np.asarray(e["ipid"])[rs], jnp.int32)
    log, counts, ofl = cascade_nucleus(ppi, pN, ppid, ipid, Npid, cfg, key, channel="res", log_cap=LCAP)
    return first_chan_fracs({k: np.asarray(v) for k, v in log.items()}, counts, rw[rs])


def main():
    tg = resolve_targets(MAT)[0][0]
    key = jax.random.PRNGKey(7)
    PN = {211: "pi+", 111: "pi0", -211: "pi-", 2212: "p", 2112: "n"}
    CH = {0: "transmit", 1: "elastic", 2: "cex/inel", 3: "abs", 4: "conv"}
    print(f"[_KSLAB A/B] {MAT} A={tg.A} N={N}  fast_xsec(K=3) vs exact(all nucleons)")
    fast = run(tg, True, key); exact = run(tg, False, key)
    worst = 0.0
    for pid in fast:
        if pid not in exact:
            continue
        line = f"  {PN[pid]:>4}: "
        for c in range(5):
            a = fast[pid].get(c, 0.0); b = exact[pid].get(c, 0.0)
            if a < 1e-4 and b < 1e-4:
                continue
            d = abs(a - b)
            worst = max(worst, d)
            line += f"{CH[c]} {a:.4f}/{b:.4f}(d{d*100:+.2f}%)  "
        print(line)
    print(f"\n  WORST |fast-exact| channel-fraction diff = {worst*100:.2f}%  "
          f"({'OK <=1%' if worst <= 0.01 else 'EXCEEDS 1% -> K=3 truncation matters'})")


if __name__ == "__main__":
    main()
