"""FULL-pool-cascade debug dump for the CC1pi tension study (ADoNIS side).

Unlike the single-pass scripts/cascade_pion_fate_dump.py (primary pion alone, no daughter re-cascade),
this runs the COMPLETE pooled cascade (the exact engine the matrix uses: setup_nucleus + make_pool_stepper
+ run_cascade_pool, RES gen-0 = primary pion + recoil nucleon) on a large C RES sample and records, PER
EVENT, everything needed to localize the CC1pi ADoNIS-vs-ACHILLES tension in INITIAL-pion-|p| brackets:

  primary pion : p_init, charge_init, FATE (absorbed / converted / survived-same / charge-exchange)
  final-state  : full surviving-pion multiplicity by charge npip/npi0/npim  (mirrors ACHILLES FATE_FS)
  proton leg   : # in-window protons + leading-proton |p|  (the CC1pi proton selection)
  + escaped-nucleon multiplicity (p/n).

BUFFER SAFETY (the point of this study): runs with LARGE fixed buffers (P=stack width, M_out=escaped
buffer, ADONIS_N_RECOIL=top-K knockouts, max_steps=march cap) and reports the stack/out OVERFLOW counters.
Created particles share the GLOBAL max_steps clock (born at step k -> max_steps-k steps left), so a too-small
max_steps drops still-inside particles SILENTLY (separate from overflow); run with --maxsteps2 to get a
saturation cross-check (identical fractions at a larger cap => no truncation).

Run: python -u scripts/cascade_debug_dump.py [n_res] [out.npz] [P] [M_out] [max_steps] [n_recoil]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

n_res    = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ado_cascade_debug_C.npz"
P_BUF    = int(sys.argv[3]) if len(sys.argv) > 3 else 16          # pool stack width M
M_OUT    = int(sys.argv[4]) if len(sys.argv) > 4 else 48          # escaped-terminal buffer
MAX_STEPS_REQ = int(sys.argv[5]) if len(sys.argv) > 5 else 1000   # march cap (floor; nucleus-scaled below)
N_RECOIL = sys.argv[6] if len(sys.argv) > 6 else "8"             # top-K knockouts per nucleon step (env)
os.environ["ADONIS_N_RECOIL"] = N_RECOIL                          # MUST precede the cascade import

import numpy as np, jax, jax.numpy as jnp
np.seterr(all="ignore")
from adonis.workflow.materials import resolve_targets
from adonis.xsec import res_xsec
from adonis.xsec.spectral import SpectralFunction
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _load_density
from adonis.fsi.cascade_full import (setup_nucleus, empty_batch, make_pool_stepper, run_cascade_pool,
                                      PION, NUCLEON, _ORIG_PRIM_PI, FATE_ESCAPE, FATE_ABSORB, FATE_CONVERT)

# CC1pi+Np tight signal windows (== cc1pi_fig_tki)
MU_LO, MU_HI = 250.0, 7000.0
PI_LO, PI_HI = 150.0, 1200.0
P_LO, P_HI = 450.0, 1200.0
COS70 = float(np.cos(np.deg2rad(70.0)))
SEEDS = 8
CHUNK = 25000                                                     # events per pool call (bounds peak memory: ~chunk*P*steps)
PI_EDGES = np.array([0, 100, 150, 200, 250, 300, 350, 400, 500, 700, 2000.0])


def build_cfg():
    tg = resolve_targets("C")[0][0]
    _, _, _, radius = _load_density(tg.density_p, tg.density_n)
    ms = max(MAX_STEPS_REQ, int(np.ceil(3.0 * radius / 0.04)))
    cfg = DiscreteCascadeConfig(step=0.04, max_steps=ms, seed=1, nn_inelastic=True, pauli=True,
                                nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
    return tg, cfg, ms, radius


def run_pool_chunk(p_pi, p_N, ppid, ipid, Npid, cfg, key):
    """Replicate _cascade_pool's RES path verbatim, but return the FULL escaped-terminal batch `out`
    (with species+charge) + overflow, so we can count final-state pions by charge."""
    su = setup_nucleus(p_pi, ppid.astype(jnp.int32), ipid.astype(jnp.int32), cfg, key)
    n = p_pi.shape[0]
    _kpi, knuc, _kpi2 = jax.random.split(su["kp"], 3)
    g0 = empty_batch(n, 2)
    g0["alive"] = jnp.ones((n, 2), bool)
    g0["species"] = jnp.array([PION, NUCLEON], jnp.int32)[None, :] * jnp.ones((n, 1), jnp.int32)
    g0["charge"] = jnp.stack([su["ch0"], (Npid == 2212).astype(jnp.int32)], axis=1)
    g0["p4"] = jnp.stack([p_pi, p_N], axis=1)
    g0["pos"] = jnp.broadcast_to(su["pos0"][:, None, :], (n, 2, 3))
    g0["origin"] = jnp.array([_ORIG_PRIM_PI, 0], jnp.int32)[None, :] * jnp.ones((n, 1), jnp.int32)
    stepper = make_pool_stepper(su, cfg, with_rec=False)
    out, sofl, oofl, prim_fate = run_cascade_pool(
        g0, stepper, knuc, su["consumed0"], M=P_BUF, max_steps=cfg.max_steps, M_out=M_OUT,
        prim_origin=_ORIG_PRIM_PI)
    return out, int(sofl), int(oofl), np.asarray(prim_fate)


def main():
    tg, cfg, ms, radius = build_cfg()
    print(f"[cfg] C  radius={radius:.2f} fm  max_steps={ms} (march {ms*0.04:.0f} fm)  "
          f"P={P_BUF} M_out={M_OUT} N_RECOIL={N_RECOIL}  SEEDS={SEEDS} n_res/seed={n_res}", flush=True)
    sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)
    cols = {k: [] for k in ("p_init", "ch_init", "prim_fate", "prim_fch", "prim_nsc", "npip", "npi0", "npim",
                            "n_inwin_p", "lead_p", "n_p_esc", "n_n_esc", "w")}
    tot_sofl = tot_oofl = 0
    for sd in range(SEEDS):
        e = res_xsec.generate(n_res, seed=sd, return_events=True, sf_n=sf_n, sf_p=sf_p,
                              n_neutron=tg.A - tg.Z, n_proton=tg.Z)["events"]
        w = np.asarray(e["w"]); sel = w > 0
        ppi = np.asarray(e["p_pi"])[sel]; pN = np.asarray(e["p_N"])[sel]
        ppid = np.asarray(e["ppid"])[sel]; ipid = np.asarray(e["ipid"])[sel]; Npid = np.asarray(e["Npid"])[sel]
        w = w[sel]; m = len(w)
        for c0 in range(0, m, CHUNK):
            sl = slice(c0, c0 + CHUNK)
            out, sofl, oofl, pfate = run_pool_chunk(
                jnp.asarray(ppi[sl]), jnp.asarray(pN[sl]), jnp.asarray(ppid[sl], jnp.int32),
                jnp.asarray(ipid[sl], jnp.int32), jnp.asarray(Npid[sl], jnp.int32),
                cfg, jax.random.PRNGKey(11 + sd))
            tot_sofl += sofl; tot_oofl += oofl
            sp = np.asarray(out["species"]); chg = np.asarray(out["charge"]); al = np.asarray(out["alive"])
            p4 = np.asarray(out["p4"]); org = np.asarray(out["origin"])
            is_pi = (sp == PION) & al
            is_p = (sp == NUCLEON) & al & (chg == 1)
            is_n = (sp == NUCLEON) & al & (chg == 0)
            cols["npip"].append((is_pi & (chg == 0)).sum(1))
            cols["npi0"].append((is_pi & (chg == 1)).sum(1))
            cols["npim"].append((is_pi & (chg == 2)).sum(1))
            cols["n_p_esc"].append(is_p.sum(1)); cols["n_n_esc"].append(is_n.sum(1))
            # primary pion final charge (if it escaped): the PION slot with origin==prim tag
            prim_pi = is_pi & (org == _ORIG_PRIM_PI)
            jj = np.argmax(prim_pi, axis=1); has_prim = prim_pi.any(1)
            prim_fch = np.where(has_prim, chg[np.arange(len(jj)), jj], -1)
            nscb = np.asarray(out["nsc"])
            prim_nsc = np.where(has_prim, nscb[np.arange(len(jj)), jj], -1)   # primary-pion scatter count (escaped only)
            cols["prim_fch"].append(prim_fch); cols["prim_fate"].append(pfate); cols["prim_nsc"].append(prim_nsc)
            # proton leg: in-window protons + leading proton |p|
            pm = np.linalg.norm(p4[:, :, 1:], axis=2); cth = p4[:, :, 3] / np.clip(pm, 1e-9, None)
            inwin = is_p & (pm > P_LO) & (pm < P_HI) & (cth > COS70)
            cols["n_inwin_p"].append(inwin.sum(1))
            pmom_p = np.where(is_p, pm, 0.0); cols["lead_p"].append(pmom_p.max(1))
            cols["p_init"].append(np.linalg.norm(ppi[sl][:, 1:], axis=1))
            cols["ch_init"].append(np.asarray(jnp.where(jnp.asarray(ppid[sl]) == 211, 0,
                                    jnp.where(jnp.asarray(ppid[sl]) == -211, 2, 1))))
            cols["w"].append(w[sl])
            print(f"    seed {sd+1}/{SEEDS} chunk @{c0}: +{len(w[sl])} ev  "
                  f"(cum overflow stack={tot_sofl} out={tot_oofl})", flush=True)
        print(f"  seed {sd+1}/{SEEDS}: +{m} events  (cum overflow stack={tot_sofl} out={tot_oofl})", flush=True)
    D = {k: np.concatenate(v) for k, v in cols.items()}
    D["w"] = D["w"] / SEEDS
    np.savez(out_path, max_steps=ms, P=P_BUF, M_out=M_OUT, n_recoil=int(N_RECOIL),
             overflow_stack=tot_sofl, overflow_out=tot_oofl, **D)
    _report(D, tot_sofl, tot_oofl, ms)
    print(f"\nwrote {out_path}", flush=True)


def _report(D, sofl, oofl, ms):
    w = D["w"]; pin = D["p_init"]; ch0 = D["ch_init"]; pf = D["prim_fate"]; pfch = D["prim_fch"]
    pip = ch0 == 0
    # primary-pion FATE per initial-|p| bracket (pi+ primaries): survived-pi+ / charge-ex / absorbed / converted
    surv_same = (pf == FATE_ESCAPE) & (pfch == ch0)
    chargex   = (pf == FATE_ESCAPE) & (pfch != ch0)
    absb      = (pf == FATE_ABSORB)
    conv      = (pf == FATE_CONVERT)
    def wf(mask, b): wb = w[b]; return float((wb * mask[b]).sum() / wb.sum()) if wb.sum() > 0 else 0.0
    print(f"\n=== ADoNIS FULL-pool primary-pi+ fate by initial |p_pi| (C, weighted)  [overflow stack={sofl} out={oofl}] ===")
    print(f"{'p_in bin':>13s} {'Nevt':>8s} {'surv_pi+':>9s} {'chg_ex':>8s} {'absorbed':>9s} {'converted':>10s} "
          f"{'<npip>':>7s} {'inwin_p>=1':>11s}")
    for lo, hi in zip(PI_EDGES[:-1], PI_EDGES[1:]):
        b = pip & (pin >= lo) & (pin < hi)
        if not b.any(): continue
        wb = w[b]
        print(f"{f'[{lo:.0f},{hi:.0f})':>13s} {int(b.sum()):8d} {wf(surv_same,b):9.4f} {wf(chargex,b):8.4f} "
              f"{wf(absb,b):9.4f} {wf(conv,b):10.4f} {(wb*D['npip'][b]).sum()/wb.sum():7.3f} "
              f"{wf(D['n_inwin_p']>=1, b):11.4f}")
    print(f"overflow: stack={sofl} out={oofl}  (both must be 0 for buffer-safe; alive-at-cap proven via --maxsteps2)")


if __name__ == "__main__":
    main()
