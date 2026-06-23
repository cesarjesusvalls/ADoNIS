"""Shared pool-FSI helper: the SINGLE entry point consumers use to run the (Gaussian, differentiable)
pool cascade and pull the common quantities, replacing the retired two-stage DiscreteCascadeFSI +
DiscreteNucleonFSI `.apply` chain.  The pool runs the pion + nucleon cascade JOINTLY (cascade_nucleus),
so there is one call, one joint kind-1 FSI record, and one `pool_fsi_reweight`.

Usage:
    from adonis.fsi.pool_fsi import run_fsi, lead_proton
    out = run_fsi(p_pi, p_N, pid_pi, pid_Ni, Npid, cfg, key, channel="res", rec_caps=(32, 64))
    absorbed   = out["pterm"]["pid"] == 0          # primary pion absorbed -> CC0pi (RES only)
    lead_p     = out["lead_prot"]                  # leading (max-|p|) escaped proton (n,4); 0 if none
    created_pi = out["created"]                    # leading cascade-created surviving pion
    w_theta    = pool_fsi_reweight(out["rec"], sabs, sscat)   # kind-1 reweight (rec_caps must be set)
"""
import numpy as np
import jax.numpy as jnp
from adonis.fsi.cascade_full import cascade_nucleus, pool_fsi_reweight, NUCLEON  # noqa: F401


def lead_proton(nt):
    """Leading (global max-|p|) escaped proton from a pool nucleon-terminal batch nt -> (n,4); 0 if none."""
    p4 = nt["p4"]; isp = (nt["pid"] == 2212) & nt["alive"]
    mom = jnp.linalg.norm(p4[:, :, 1:], axis=2) * isp
    n = p4.shape[0]; ar = jnp.arange(n); j = jnp.argmax(mom, axis=1)
    return jnp.where((mom[ar, j] > 0)[:, None], p4[ar, j], 0.0)


def proton_candidates(nterms, mprot):
    """Top-`mprot` escaped PROTON terminals across ALL cascade generations ->
    (prot (n,mprot,4), prot_origin (n,mprot), prot_gen (n,mprot)); non-proton/empty slots are 0 / -1.
    Verbatim of adonis.workflow.generate.run_one_seed's prot[] construction -- the single validated
    pool proton-candidate set (origin: 0=RES/QE nucleon chain, 1=pion-knockout, 2=primary pion)."""
    p4s, orgs, gns = [], [], []
    for g in nterms:
        sp = np.asarray(g["species"]); pid = np.asarray(g["pid"]); p4 = np.asarray(g["p4"]); al = np.asarray(g["alive"])
        good = (sp == NUCLEON) & (pid == 2212) & al
        p4s.append(np.where(good[:, :, None], p4, 0.0))
        orgs.append(np.where(good, np.asarray(g["origin"]), -1))
        gns.append(np.where(good, np.asarray(g["gen"]), -1))
    P4 = np.concatenate(p4s, axis=1); ORG = np.concatenate(orgs, axis=1); GN = np.concatenate(gns, axis=1)
    n = P4.shape[0]; ar = np.arange(n)
    mom = np.linalg.norm(P4[:, :, 1:], axis=2)
    idx = np.argsort(-mom, axis=1)[:, :mprot]; g2 = ar[:, None]
    return P4[g2, idx], ORG[g2, idx].astype(np.int64), GN[g2, idx].astype(np.int64)


def run_fsi(p_pi, p_N, pid_pi, pid_Ni, Npid, cfg, key, channel="res", P=16, max_gen=6, rec_caps=None):
    """Run the pool cascade and return the common per-event FSI outputs as a dict:
      pterm   : primary-pion terminal (RES: pid 0=absorbed, -1=converted, else surviving-pi pid; QE: 0)
      nterms  : list of per-generation nucleon-terminal batches (escaped protons live here)
      created : leading cascade-created surviving pion (p4,pid,alive)
      ofl     : pool stack/out overflow count (should be ~0)
      lead_prot: leading escaped proton (n,4)
      rec     : joint kind-1 FSI record (None unless rec_caps set) -> feed pool_fsi_reweight
    Engine is always the pool (cfg.engine defaults to 'pool')."""
    res = cascade_nucleus(p_pi, p_N, jnp.asarray(pid_pi, jnp.int32), jnp.asarray(pid_Ni, jnp.int32),
                         jnp.asarray(Npid, jnp.int32), cfg, key, P=P, max_gen=max_gen,
                         channel=channel, rec_caps=rec_caps)
    pterm, nterms, ofl, created = res[:4]
    rec = res[4] if len(res) == 5 else None
    return dict(pterm=pterm, nterms=nterms, ofl=ofl, created=created,
                lead_prot=lead_proton(nterms[0]), rec=rec)
