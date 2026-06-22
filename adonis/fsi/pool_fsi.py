"""Shared pool-FSI helper: the SINGLE entry point consumers use to run the (Gaussian, differentiable)
pool cascade and pull the common quantities, replacing the retired two-stage DiscreteCascadeFSI +
DiscreteNucleonFSI `.apply` chain.  The pool runs the pion + nucleon cascade JOINTLY (cascade_carbon),
so there is one call, one joint kind-1 FSI record, and one `pool_fsi_reweight`.

Usage:
    from adonis.fsi.pool_fsi import run_fsi, lead_proton
    out = run_fsi(p_pi, p_N, pid_pi, pid_Ni, Npid, cfg, key, channel="res", rec_caps=(32, 64))
    absorbed   = out["pterm"]["pid"] == 0          # primary pion absorbed -> CC0pi (RES only)
    lead_p     = out["lead_prot"]                  # leading (max-|p|) escaped proton (n,4); 0 if none
    created_pi = out["created"]                    # leading cascade-created surviving pion
    w_theta    = pool_fsi_reweight(out["rec"], sabs, sscat)   # kind-1 reweight (rec_caps must be set)
"""
import jax.numpy as jnp
from adonis.fsi.cascade_full import cascade_carbon_v2 as cascade_carbon, pool_fsi_reweight  # noqa: F401


def lead_proton(nt):
    """Leading (global max-|p|) escaped proton from a pool nucleon-terminal batch nt -> (n,4); 0 if none."""
    p4 = nt["p4"]; isp = (nt["pid"] == 2212) & nt["alive"]
    mom = jnp.linalg.norm(p4[:, :, 1:], axis=2) * isp
    n = p4.shape[0]; ar = jnp.arange(n); j = jnp.argmax(mom, axis=1)
    return jnp.where((mom[ar, j] > 0)[:, None], p4[ar, j], 0.0)


def run_fsi(p_pi, p_N, pid_pi, pid_Ni, Npid, cfg, key, channel="res", P=12, max_gen=6, rec_caps=None):
    """Run the pool cascade and return the common per-event FSI outputs as a dict:
      pterm   : primary-pion terminal (RES: pid 0=absorbed, -1=converted, else surviving-pi pid; QE: 0)
      nterms  : list of per-generation nucleon-terminal batches (escaped protons live here)
      created : leading cascade-created surviving pion (p4,pid,alive)
      ofl     : pool stack/out overflow count (should be ~0)
      lead_prot: leading escaped proton (n,4)
      rec     : joint kind-1 FSI record (None unless rec_caps set) -> feed pool_fsi_reweight
    Engine is always the pool (cfg.engine defaults to 'pool')."""
    res = cascade_carbon(p_pi, p_N, jnp.asarray(pid_pi, jnp.int32), jnp.asarray(pid_Ni, jnp.int32),
                         jnp.asarray(Npid, jnp.int32), cfg, key, P=P, max_gen=max_gen,
                         channel=channel, rec_caps=rec_caps)
    pterm, nterms, ofl, created = res[:4]
    rec = res[4] if len(res) == 5 else None
    return dict(pterm=pterm, nterms=nterms, ofl=ofl, created=created,
                lead_prot=lead_proton(nterms[0]), rec=rec)
