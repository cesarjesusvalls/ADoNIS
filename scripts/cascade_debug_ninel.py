"""FULLY-RESOLVED NN->NDelta->Npi probe (ADoNIS): single-pass primary nucleon, recording per FIRST
inelastic interaction the struck-background species and the created-pion charge -- so the created-pion
rate splits by NUCLEON-PAIR type (pp/pn/nn) AND PION type (pi+/pi0/pi-), per primary |p| bracket.

Faithfully replicates the adonis.fsi.cascade_discrete._nucleon_step selection (Gaussian-roll closest
passer over the in-slab background) + the inelastic branch (chose_inel; Delta charge dch; pion charge
pi_q; two-nucleon Pauli in_blocked).  Single-pass = primary only (the QE created-pi+ is dominated by the
primary proton's first inelastic interaction); validate the TOTAL pi+/pi0/pi- rate against the full-pool
QE dump (cascade_debug_qe.py).

Run: python -u scripts/cascade_debug_ninel.py [n_qe_per_seed] [primary p|n] [out.npz]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ADONIS_N_RECOIL", "8")
import numpy as np, jax, jax.numpy as jnp
np.seterr(all="ignore")
from adonis.workflow.materials import resolve_targets
from adonis.xsec import qe_xsec
from adonis.xsec.spectral import SpectralFunction
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
from adonis.fsi.cascade_discrete import (DiscreteCascadeConfig, _load_density, _rho_species, _kf_local,
                                         nn_elastic_sigma, MB_TO_FM2, M_N)
from adonis.fsi import nn_inelastic as nni
from adonis.fsi.cascade_full import setup_nucleus

n_qe = int(sys.argv[1]) if len(sys.argv) > 1 else 120000
PRIM = sys.argv[2] if len(sys.argv) > 2 else "p"          # primary nucleon species
out_path = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/ado_ninel_{PRIM}.npz"
SEEDS = 8; ED = [0, 700, 800, 900, 1000, 1200, 1500, 3000.0]


def step_record(p4, pos, dhat, fz, isp, alive, consumed, npos, nmom, nisp, rgrid, rhoP, rhoN, radius, cfg, key):
    """One _nucleon_step, returning the standard kinematic update PLUS per-event diagnostics:
    (made_pi, pi_q in {1,0,-1}, struck_isp in {1 p,0 n,-1 none}, qpair)."""
    n, A = nisp.shape; ar = jnp.arange(n)
    outward = jnp.sum(pos * dhat, axis=1) > 0
    escaping = (jnp.linalg.norm(pos, axis=1) > radius) & outward
    alive = alive & ~escaping
    can_int = fz <= 0.0
    rel = npos - pos[:, None, :]; par = jnp.sum(rel * dhat[:, None, :], axis=2)
    perp2 = jnp.sum(rel ** 2, axis=2) - par ** 2
    in_slab = (par > 0) & (par <= cfg.step) & (~consumed) & alive[:, None]
    Pp = p4[:, None, :] + nmom
    s = Pp[:, :, 0] ** 2 - jnp.sum(Pp[:, :, 1:] ** 2, axis=2)
    sqrts = jnp.sqrt(jnp.clip(s, (2 * M_N) ** 2, None))
    same_iso = isp[:, None] == nisp
    sig_el = jnp.clip(nn_elastic_sigma(sqrts, same_iso), 0.0, None)
    pcm = jnp.sqrt(jnp.clip(s / 4.0 - M_N ** 2, 1e-6, None)) / 1000.0
    sig_in = jnp.clip(nni.sigma_nn_ndelta(sqrts / 1000.0, pcm, same_iso), 0.0, None)
    sig = sig_el + sig_in
    prob = jnp.where(in_slab, jnp.exp(-jnp.pi * perp2 / jnp.clip(sig * MB_TO_FM2, 1e-12, None)), 0.0)
    sk, ku, ks = jax.random.split(key, 3)
    passes = in_slab & (jax.random.uniform(ku, (n, A)) < prob)
    big = jnp.where(passes, perp2, jnp.inf); j = jnp.argmin(big, axis=1)
    has_hit = jnp.isfinite(big[ar, j]) & alive & can_int
    struck = nisp[ar, j].astype(jnp.int32)                                  # 1 proton, 0 neutron
    pN_j = nmom[ar, j]                                                       # struck nucleon 4-momentum
    sig_in_j = sig_in[ar, j]; sig_el_j = sig_el[ar, j]
    u_br = jax.random.uniform(jax.random.fold_in(sk, 101), (n,))
    chose_inel = has_hit & (u_br < sig_in_j / jnp.clip(sig_el_j + sig_in_j, 1e-12, None))
    q_pair = isp.astype(jnp.int32) + struck
    u107 = jax.random.uniform(jax.random.fold_in(sk, 107), (n,))
    u108 = jax.random.uniform(jax.random.fold_in(sk, 108), (n,))
    dch = jnp.where(q_pair == 2, jnp.where(u107 < 0.75, 2, 1),
            jnp.where(q_pair == 1, jnp.where(u107 < 0.5, 1, 0),
                                   jnp.where(u107 < 0.25, 0, -1)))
    pi_q = jnp.where(dch == 2, 1, jnp.where(dch == 1, jnp.where(u108 < 1.0 / 3.0, 1, 0),
            jnp.where(dch == 0, jnp.where(u108 < 2.0 / 3.0, 0, -1), -1)))
    # ----- inelastic Pauli block (faithful to _nucleon_step): the two outgoing nucleons must clear k_F
    Pj = p4 + pN_j
    rs_j = jnp.sqrt(jnp.clip(Pj[:, 0] ** 2 - jnp.sum(Pj[:, 1:] ** 2, axis=1), (2 * M_N) ** 2, None))
    u_m = jax.random.uniform(jax.random.fold_in(sk, 102), (n,))
    m_d = jnp.clip(nni.sample_delta_mass(rs_j / 1000.0, u_m) * 1000.0, M_N + 135.0, rs_j - M_N - 1.0)
    cth1 = 2 * jax.random.uniform(jax.random.fold_in(sk, 103), (n,)) - 1.0
    phi1 = 2 * jnp.pi * jax.random.uniform(jax.random.fold_in(sk, 104), (n,))
    cth2 = 2 * jax.random.uniform(jax.random.fold_in(sk, 105), (n,)) - 1.0
    phi2 = 2 * jnp.pi * jax.random.uniform(jax.random.fold_in(sk, 106), (n,))

    def _split2(P4, mA, mB, cth_, phi_):
        ss = jnp.clip(P4[:, 0] ** 2 - jnp.sum(P4[:, 1:] ** 2, axis=1), (mA + mB) ** 2 * 1.0001, None)
        rss = jnp.sqrt(ss); EA = (ss + mA ** 2 - mB ** 2) / (2 * rss)
        pf = jnp.sqrt(jnp.clip(EA ** 2 - mA ** 2, 0.0, None)); sth_ = jnp.sqrt(jnp.clip(1 - cth_ ** 2, 0, None))
        d_ = jnp.stack([sth_ * jnp.cos(phi_), sth_ * jnp.sin(phi_), cth_], axis=1)
        pa = jnp.concatenate([EA[:, None], pf[:, None] * d_], axis=1)
        pb = jnp.concatenate([(rss - EA)[:, None], -pf[:, None] * d_], axis=1)
        beta_ = P4[:, 1:] / P4[:, [0]]; b2_ = jnp.sum(beta_ ** 2, axis=1)
        g_ = 1 / jnp.sqrt(jnp.clip(1 - b2_, 1e-12, None))
        def lab(p4_):
            bp_ = jnp.sum(beta_ * p4_[:, 1:], axis=1); E = g_ * (p4_[:, 0] + bp_)
            p3 = p4_[:, 1:] + ((g_ - 1) * bp_ / jnp.clip(b2_, 1e-30, None) + g_ * p4_[:, 0])[:, None] * beta_
            return jnp.concatenate([E[:, None], p3], axis=1)
        return lab(pa), lab(pb)

    pN1, pD = _split2(Pj, jnp.full((n,), M_N), m_d, cth1, phi1)
    pN2, _pPiX = _split2(pD, jnp.full((n,), M_N), jnp.full((n,), 138.04), cth2, phi2)
    _rnuc_j = jnp.linalg.norm(npos[ar, j], axis=1); _r_lead = jnp.linalg.norm(pos, axis=1)
    kf_p_j = _kf_local(_rho_species(_rnuc_j, rgrid, rhoP)); kf_n_j = _kf_local(_rho_species(_rnuc_j, rgrid, rhoN))
    _kfp_lead = _kf_local(_rho_species(_r_lead, rgrid, rhoP)); _kfn_lead = _kf_local(_rho_species(_r_lead, rgrid, rhoN))
    nl_is1 = jnp.linalg.norm(pN1[:, 1:], axis=1) >= jnp.linalg.norm(pN2[:, 1:], axis=1)
    _n1p = (q_pair - dch) == 1; _n2p = (dch - pi_q) == 1
    kf_N1 = jnp.where(nl_is1, jnp.where(_n1p, _kfp_lead, _kfn_lead), jnp.where(_n1p, kf_p_j, kf_n_j))
    kf_N2 = jnp.where(~nl_is1, jnp.where(_n2p, _kfp_lead, _kfn_lead), jnp.where(_n2p, kf_p_j, kf_n_j))
    in_blocked = ((jnp.linalg.norm(pN1[:, 1:], axis=1) < kf_N1) | (jnp.linalg.norm(pN2[:, 1:], axis=1) < kf_N2))
    if not cfg.pauli:
        in_blocked = in_blocked & False
    made = chose_inel & ~in_blocked
    pos2 = pos + cfg.step * dhat
    fz2 = jnp.clip(fz - cfg.step, 0.0, None)
    consumed2 = consumed | (jax.nn.one_hot(j, A).astype(bool) & has_hit[:, None])
    return (p4, pos2, dhat, fz2, alive), made, pi_q, jnp.where(has_hit, struck, -1), q_pair, consumed2


def main():
    tg = resolve_targets("C")[0][0]
    _, _, _, radius = _load_density(tg.density_p, tg.density_n)
    ms = max(1000, int(np.ceil(3.0 * radius / 0.04)))
    cfg = DiscreteCascadeConfig(step=0.04, max_steps=ms, seed=1, nn_inelastic=True, pauli=True,
                                nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
    rgrid, rhoP, rhoN, rad = _load_density(tg.density_p, tg.density_n)
    sf_n = SpectralFunction(tg.spectral_n)
    print(f"[cfg] C  primary={PRIM}  max_steps={ms}  SEEDS={SEEDS} n_qe/seed={n_qe}", flush=True)
    cols = {k: [] for k in ("p_init", "made_pip", "made_pi0", "made_pim", "struck", "w")}
    for sd in range(SEEDS):
        r = qe_xsec.sample_importance(n_qe, seed=sd, sf=sf_n, n_neutron=tg.A - tg.Z)
        pN = jnp.asarray(np.asarray(r["p_out"])); w = np.asarray(r["w"]) / n_qe; m = len(w)
        su = setup_nucleus(pN, jnp.zeros(m, jnp.int32), jnp.full(m, 2112, jnp.int32), cfg, jax.random.PRNGKey(11 + sd))
        npos, nmom, nisp = su["npos"], su["nmom"], su["nisp"]
        p4 = pN; pos = su["pos0"]; d3 = p4[:, 1:]
        dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
        fz = jnp.zeros(m); alive = jnp.ones(m, bool); consumed = su["consumed0"]
        isp = jnp.full(m, PRIM == "p")
        keys = jax.random.split(jax.random.PRNGKey(99 + sd), ms)
        made_pip = np.zeros(m, bool); made_pi0 = np.zeros(m, bool); made_pim = np.zeros(m, bool)
        first_struck = np.full(m, -1, np.int32)
        for i in range(ms):
            (p4, pos, dhat, fz, alive), made, pi_q, struck, qpair, consumed = step_record(
                p4, pos, dhat, fz, isp, alive, consumed, npos, nmom, nisp, rgrid, rhoP, rhoN, rad, cfg, keys[i])
            made = np.asarray(made); piq = np.asarray(pi_q); st = np.asarray(struck)
            new = made & ~(made_pip | made_pi0 | made_pim)                  # FIRST creation only
            made_pip |= new & (piq == 1); made_pi0 |= new & (piq == 0); made_pim |= new & (piq == -1)
            first_struck = np.where(new, st, first_struck)
        cols["p_init"].append(np.linalg.norm(np.asarray(pN)[:, 1:], axis=1))
        cols["made_pip"].append(made_pip); cols["made_pi0"].append(made_pi0); cols["made_pim"].append(made_pim)
        cols["struck"].append(first_struck); cols["w"].append(w)
        print(f"  seed {sd+1}/{SEEDS}: pi+={made_pip.mean():.4f} pi0={made_pi0.mean():.4f} pi-={made_pim.mean():.4f}", flush=True)
    D = {k: np.concatenate(v) for k, v in cols.items()}; D["w"] = D["w"] / SEEDS
    np.savez(out_path, primary=PRIM, **D)
    w = D["w"]; pin = D["p_init"]; st = D["struck"]
    pair = {1: "pp" if PRIM == "p" else "pn", 0: "pn" if PRIM == "p" else "nn"}
    print(f"\n=== ADoNIS NN-inelastic created-pion rate, primary={PRIM}, per p|p| x background x pion ===")
    print(f"overall: pi+={float((w*D['made_pip']).sum()/w.sum()):.4f} pi0={float((w*D['made_pi0']).sum()/w.sum()):.4f} "
          f"pi-={float((w*D['made_pim']).sum()/w.sum()):.4f}")
    for sp, nm in ((1, pair[1]), (0, pair[0])):
        print(f"\n-- struck background = {('proton' if sp==1 else 'neutron')} (pair {nm}) --")
        print(f"{'p_N':>12}{'N_inel':>8}{'pi+':>9}{'pi0':>9}{'pi-':>9}")
        for lo, hi in zip(ED[:-1], ED[1:]):
            b = (pin >= lo) & (pin < hi) & (st == sp); wb = w[(pin >= lo) & (pin < hi)]
            wall = w[(pin >= lo) & (pin < hi)]
            if wall.sum() <= 0: continue
            sel = (pin >= lo) & (pin < hi)
            num_pip = float((w[sel] * (D['made_pip'][sel] & (st[sel] == sp))).sum() / wall.sum())
            num_pi0 = float((w[sel] * (D['made_pi0'][sel] & (st[sel] == sp))).sum() / wall.sum())
            num_pim = float((w[sel] * (D['made_pim'][sel] & (st[sel] == sp))).sum() / wall.sum())
            nn = int(((pin >= lo) & (pin < hi) & (st == sp)).sum())
            print(f"{f'[{lo:.0f},{hi:.0f})':>12}{nn:8d}{num_pip:9.4f}{num_pi0:9.4f}{num_pim:9.4f}")
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
