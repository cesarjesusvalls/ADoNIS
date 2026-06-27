"""Stage-0 golden reference for the persistent-refill engine rewrite (docs/logbook/
cascade_persistent_refill_plan.md).  Runs the CURRENT cascade engine on a FIXED C sample (RES+QE) in all
three modes (forward rich schema / with_rec FSI records / with_seg segment log) and saves every output to
/tmp/refeng_C.npz.  After each rewrite stage, scripts/_engine_check.py compares against this -> any
difference beyond the intended kept-overflow-particles is a regression.

Run: python -u scripts/_engine_golden.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

N = 3000; KEY = jax.random.PRNGKey(20240623)


def sample(tg):
    e = res_xsec.generate(N, seed=0, return_events=True, sf_n=SpectralFunction(tg.spectral_n),
                          sf_p=SpectralFunction(tg.spectral_p), n_neutron=tg.A - tg.Z, n_proton=tg.Z)["events"]
    rs = np.asarray(e["w"]) > 0
    res = dict(ppi=jnp.asarray(np.asarray(e["p_pi"])[rs]), ppid=jnp.asarray(np.asarray(e["ppid"])[rs], jnp.int32),
               pN=jnp.asarray(np.asarray(e["p_N"])[rs]), Npid=jnp.asarray(np.asarray(e["Npid"])[rs], jnp.int32),
               ipid=jnp.asarray(np.asarray(e["ipid"])[rs], jnp.int32))
    q = qe_xsec.sample_importance(N, seed=0, sf=SpectralFunction(tg.spectral_n), n_neutron=tg.A - tg.Z)
    m = np.asarray(q["w"]) != 0
    qe = dict(pN=jnp.asarray(np.asarray(q["p_out"])[m]), Npid=jnp.full(int(m.sum()), 2212, jnp.int32))
    return res, qe


def run_all(D, tg, n_w=0, q_cap=0):
    """Forwards (n_w, q_cap) verbatim to cascade_nucleus.  DEFAULT (0,0) = lock-step no-queue = the golden
    reference.  n_w=None,q_cap=None -> the production DEFAULT path (auto-refill + waiting-queue).  n_w=int,
    q_cap=int -> explicit refill/queue (used by _engine_refill_check to gate refill==golden)."""
    _, _, _, radius = _load_density(tg.density_p, tg.density_n)
    ms = max(1000, int(np.ceil(3.0 * radius / 0.04)))
    cfg = DiscreteCascadeConfig(step=0.04, max_steps=100000, seed=1, nn_inelastic=True, pauli=True,
                                nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
    res, qe = sample(tg)
    out = {}
    z = jnp.zeros
    for tag, ch, a in [("res", "res", res), ("qe", "qe", qe)]:
        n = a["pN"].shape[0]
        ppi = a.get("ppi", z((n, 4))); ppid = a.get("ppid", jnp.zeros(n, jnp.int32))
        ipid = a.get("ipid", jnp.full(n, 2112, jnp.int32))
        pidNi = ipid if ch == "res" else jnp.full(n, 2112, jnp.int32)
        nw = n_w; qc = q_cap                                          # forwarded verbatim (None=auto, 0=off)
        # forward rich schema
        pt, nt, ofl, cr = cascade_nucleus(ppi, a["pN"], ppid, pidNi, a["Npid"], cfg, KEY, channel=ch, n_w=nw, q_cap=qc)
        out[f"{tag}_pterm_pid"] = np.asarray(pt["pid"]); out[f"{tag}_pterm_p4"] = np.asarray(pt["p4"])
        out[f"{tag}_nt_pid"] = np.asarray(nt[0]["pid"]); out[f"{tag}_nt_p4"] = np.asarray(nt[0]["p4"])
        out[f"{tag}_nt_alive"] = np.asarray(nt[0]["alive"])
        out[f"{tag}_cr_pid"] = np.asarray(cr["pid"]); out[f"{tag}_cr_p4"] = np.asarray(cr["p4"])
        out[f"{tag}_ofl"] = np.asarray(ofl)
        # with_rec
        *_, rb = cascade_nucleus(ppi, a["pN"], ppid, pidNi, a["Npid"], cfg, KEY, channel=ch, rec_caps=(32, 64), n_w=nw, q_cap=qc)
        for kk, vv in rb.items():                                # rb = the per-event FSI record dict
            out[f"{tag}_rec_{kk}"] = np.asarray(vv)
        # with_seg
        log, counts, logofl = cascade_nucleus(ppi, a["pN"], ppid, pidNi, a["Npid"], cfg, KEY, channel=ch, log_cap=64, n_w=nw, q_cap=qc)
        for kk, vv in log.items():
            out[f"{tag}_log_{kk}"] = np.asarray(vv)
        out[f"{tag}_counts"] = np.asarray(counts); out[f"{tag}_logofl"] = np.asarray(logofl)
    return out


def main():
    tg = resolve_targets("C")[0][0]
    out = run_all(None, tg)
    np.savez("/tmp/refeng_C.npz", **out)
    print(f"wrote /tmp/refeng_C.npz  ({len(out)} arrays)")
    print(f"  res nterms shape {out['res_nt_pid'].shape}  ofl {int(out['res_ofl'])}  logofl {int(out['res_logofl'])}")
    print(f"  qe  nterms shape {out['qe_nt_pid'].shape}  ofl {int(out['qe_ofl'])}  logofl {int(out['qe_logofl'])}")


if __name__ == "__main__":
    main()
