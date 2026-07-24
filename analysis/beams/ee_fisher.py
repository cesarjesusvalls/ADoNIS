"""KNOB x SAMPLE Fisher: what does the inclusive (e,e') QE sample add on top of T2K?

The (e,e') sample is a PURELY VECTOR, NO-FLUX probe of the SAME QE hard vertex + spectral function as
T2K CC0pi.  So it responds to vector_strength / mu_p / mu_n / gep / gen / sf_norm / src_tail / kF_sf /
Eb_shift / qe_norm -- but is BLIND to axial_strength / M_A_qe (no axial current) and to every FSI knob
(no cascade).  That asymmetry is the point: it can break the rate-only flat direction where qe_norm,
axial_strength, vector_strength and sf_norm are four ways to spell "scale the QE rate" that no amount of
T2K data separates (docs/logbook/electron_scattering.md).

Fisher is ADDITIVE: F_total = F_T2K + F_eC, F_s = J_s^T C_s^-1 J_s.  The (e,e') J is one jax.jvp per knob
through the EXACT reweight -- recompute the EM matrix element under (vector_strength, mu_p/mu_n/gep/gen)
and the spectral function under (sf_norm, src_tail, kF_sf, Eb_shift), the SAME knob definitions the T2K
bank_reweight uses -- binned in dsigma/domega inside the theta_e' acceptance.

Usage:  python -m analysis.beams.ee_fisher [n_per_species] [--syst 0.05] [--nbins 20]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "altgen"))


def ee_jacobian(n_per_species=1_500_000, material="C", nbins=20, syst=0.05, log=print):
    """(J (nbins, NPAR), sigma (nbins,), central, edges) for the (e,e') QE dsigma/domega sample."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from adonis.xsec import ee_xsec as EE
    from adonis.xsec.backend import me_cross_section
    from adonis.analysis.sf_reweight import sf_grids, sf_reweight, removal_from_struck
    from adonis.xsec.spectral import SpectralFunction
    from analysis.t2k.differentiability.full_knobs import nominal_knobs
    import physical_fit as PF

    nom = nominal_knobs()
    knom = PF.knobs_of(jnp.asarray(PF.theta_nominal(nom)), nom)          # SAME knobs W(theta_nom) uses
    sf_nom = dict(kF_sf=float(knom["kF_sf"]), Eb_shift=float(knom["Eb_shift"]),
                  sf_norm=float(knom["sf_norm"]), src_tail=float(knom["src_tail"]))
    R = EE.generate(n_per_species, material=material, records=True)      # theta-accepted + kinematics
    _, _, sf_p_path, sf_n_path = EE.MATERIALS[material]
    grids_p = sf_grids(SpectralFunction(sf_p_path)); grids_n = sf_grids(SpectralFunction(sf_n_path))
    # Drop the handful of SF-grid-EDGE events whose nominal SF reweight or matrix element is non-finite
    # or zero (extreme struck |p|/E_removal the grid interpolation can't reach at the nominal Eb_shift):
    # negligible weight (c ~ 0) but they poison me_new/me_nom.  Evaluate sf_reweight at the ACTUAL nominal
    # knobs (Eb_shift=0.01, not the 0.0 default) so the mask matches W(theta_nom).  Guarantees w0 == 1.
    ps0 = jnp.asarray(R["p_struck"]); isp0 = np.asarray(R["is_p"])
    pm0, er0 = removal_from_struck(ps0)
    sfw0 = np.where(isp0, np.asarray(sf_reweight(grids_p, pm0, er0, **sf_nom)),
                    np.asarray(sf_reweight(grids_n, pm0, er0, **sf_nom)))
    good = np.isfinite(sfw0) & (sfw0 > 0) & np.isfinite(R["me"]) & (np.asarray(R["me"]) != 0)
    if not good.all():
        log(f"  [e-C] dropped {int((~good).sum())} SF-edge events (c-sum {R['c'][~good].sum():.2e} nb)")
    for kk in ("c", "omega", "theta", "is_p", "k_e", "k_e_out", "p_struck", "p_out", "me", "had_mass"):
        R[kk] = R[kk][good]
    log(f"  [e-C] {len(R['c']):,} accepted (e,e') events ({material}), sigma_acc={R['c'].sum():.1f} nb")
    k_e = jnp.asarray(R["k_e"]); k_e_out = jnp.asarray(R["k_e_out"])
    p_struck = jnp.asarray(R["p_struck"]); p_out = jnp.asarray(R["p_out"])
    is_p = jnp.asarray(R["is_p"]); had_mass = jnp.asarray(R["had_mass"]); me_nom = jnp.asarray(R["me"])
    c_nom = np.asarray(R["c"]); omega = np.asarray(R["omega"])
    pmag, erem = removal_from_struck(p_struck)

    def W(theta):
        k = PF.knobs_of(theta, nom)
        ff = {"gep": k["gep"], "gen": k["gen"], "gmp": k["mu_p"], "gmn": k["mu_n"]}
        d = me_cross_section(k_e, k_e_out, p_struck, p_out, spin_avg=0.25, had_mass=had_mass,
                             probe="EM", is_proton=is_p, vector_scale=k["vector_strength"], ff_scale=ff)
        me_ratio = d["me_xsec"] / me_nom
        sf_kw = dict(kF_sf=k["kF_sf"], Eb_shift=k["Eb_shift"], sf_norm=k["sf_norm"], src_tail=k["src_tail"])
        sfw = jnp.where(is_p, sf_reweight(grids_p, pmag, erem, **sf_kw),
                        sf_reweight(grids_n, pmag, erem, **sf_kw))
        return me_ratio * sfw * k["qe_norm"]

    th0 = jnp.asarray(PF.theta_nominal(nom))
    W0 = W(th0)                                                          # knob-nominal (absorbs Eb=0.01)
    def w_of(theta):
        return W(theta) / W0                                            # w(theta_nom) == 1 exactly

    w0 = np.asarray(w_of(th0))
    assert np.abs(w0 - 1).max() < 1e-9, f"(e,e') nominal identity broken: {np.abs(w0-1).max():.2e}"

    edges = np.linspace(0, 500, nbins + 1); bw = np.diff(edges)
    idx = np.digitize(omega, edges) - 1
    sel = (idx >= 0) & (idx < nbins)

    def binned(w):
        h = np.zeros(nbins)
        np.add.at(h, idx[sel], (c_nom * np.asarray(w))[sel])
        return h / bw

    central = binned(w0)
    var = np.zeros(nbins); np.add.at(var, idx[sel], (c_nom[sel]) ** 2)
    mcerr = np.sqrt(var) / bw
    sig = np.sqrt((syst * central) ** 2 + mcerr ** 2)
    empty = (central == 0) & (mcerr == 0); sig = np.where(empty, np.inf, sig)

    jvp = jax.jit(lambda th, tang: jax.jvp(w_of, (th,), (tang,))[1])
    NPAR = PF.NPAR
    J = np.zeros((nbins, NPAR))
    for kk in range(NPAR):
        g = jvp(th0, jnp.zeros(NPAR).at[kk].set(1.0))
        J[:, kk] = binned(g)
    log(f"  [e-C] {nbins} dsigma/domega bins, med MC err "
        f"{np.median(mcerr[central > 0] / central[central > 0]):.1%}")
    return J, sig, central, edges


def main(n_per_species=1_500_000, syst=0.05, nbins=20):
    import physical_fit as PF
    from analysis.paper import style
    PNAMES, PRIOR, NPAR = PF.PNAMES, PF.PRIOR, PF.NPAR

    d = np.load(style.ALTGEN / "physfit_gate1_full_v2.npz", allow_pickle=True)
    J_t2k = d["J"] / d["sigma"][:, None]
    F_t2k = J_t2k.T @ J_t2k
    Je, se, _c, _e = ee_jacobian(n_per_species, nbins=nbins, syst=syst)
    Jw = Je / se[:, None]
    F_ee = Jw.T @ Jw

    def shrink(Fm):
        V = np.linalg.inv(Fm + np.diag(1.0 / PRIOR ** 2))
        return np.sqrt(np.diag(V)) / PRIOR

    cols = [("T2K", F_t2k), ("T2K+eC", F_t2k + F_ee)]
    S = np.column_stack([shrink(Fm) for _, Fm in cols])
    print(f"\n== knob x sample Fisher: T2K vs T2K + (e,e')-C  (syst {syst:.0%}, {nbins} bins) ==")
    print(f"{'knob':>20} {'T2K':>8} {'T2K+eC':>8}   note")
    for kk in np.argsort(S[:, 0]):
        gained = "  <-- GAINED (crosses 0.5)" if S[kk, 1] < 0.5 <= S[kk, 0] else ""
        star = "".join("*" if S[kk, c] < 0.5 else " " for c in range(2))
        print(f"{PNAMES[kk]:>20} {S[kk,0]:8.2f} {S[kk,1]:8.2f} {star}{gained}")
    g = [PNAMES[kk] for kk in range(NPAR) if S[kk, 1] < 0.5 <= S[kk, 0]]
    print(f"\n  T2K: {int((S[:,0]<0.5).sum())}/{NPAR} FIT   T2K+eC: {int((S[:,1]<0.5).sum())}/{NPAR} FIT"
          + (f"   GAINED: {g}" if g else "   (no new knobs cross 0.5)"))
    np.savez("output/altgen/ee_fisher.npz", S=S, cols=["T2K", "T2K+eC"], pnames=PNAMES,
             prior=PRIOR, syst=syst, nbins=nbins, F_ee=F_ee)
    print("[out] output/altgen/ee_fisher.npz")
    return S


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n_per_species", type=int, nargs="?", default=1_500_000)
    ap.add_argument("--syst", type=float, default=0.05)
    ap.add_argument("--nbins", type=int, default=20)
    a = ap.parse_args()
    main(n_per_species=a.n_per_species, syst=a.syst, nbins=a.nbins)
