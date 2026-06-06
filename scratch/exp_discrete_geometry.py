"""DECISIVE EXPERIMENT: does the discrete-nucleon impact-parameter geometry (ACHILLES) drop
the pion-carbon absorption cross section vs the continuum mean-free-path (my cascade_real)?

Hypothesis: at the Delta sigma_tot ~ 150 mb = 15 fm^2, while carbon's transverse area is only
~23 fm^2, so sqrt(sigma/pi) ~ 2.2 fm -- a single nucleon's interaction "disk" covers much of
the nucleus.  The continuum identity int exp(-pi b^2/sigma) d^2b = sigma assumes nucleons
continuously fill an infinite plane; with only A=12 discrete nucleons and sigma ~ nuclear area,
that OVER-counts (black-disk shadowing).  If the discrete walk gives ~60 mb where the continuum
gives ~86 mb, the geometry alone explains the over-absorption.

Both estimators use the SAME cross sections (Oset abs + DCC scatter) -- only the transport
geometry differs.  numpy only (a hypothesis test, not the differentiable production cascade).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path

import adonis.fsi.oset_xsec as ox
from adonis.fsi.mb.anl_xsec import load_anl, _channel_sigma
from adonis.fsi.mb.cascade_mb import _CHANNELS

HBARC = ox.HBARC; M_N = ox.M_N; M_PIP = ox.M_PIP
MB_TO_FM2 = 0.1
rng = np.random.default_rng(1)

# ---- density rho(r) (col1 = proton = neutron number density; total = 2*col1) ---------------
d = np.loadtxt(Path(__file__).resolve().parents[1] / "data" / "nuclear" / "c12_density.txt", comments="#")
rgrid, rho_p = d[:, 0], d[:, 1]
R_NUC = float(rgrid[rho_p > 1e-4 * rho_p[0]].max())          # ~ physical boundary
def rho_proton(r):  return np.interp(r, rgrid, rho_p, left=rho_p[0], right=0.0)
def kf_local(r):    return np.cbrt(np.clip(rho_proton(r), 0, None) * 3 * np.pi**2) * HBARC

# ---- scatter sigma(W) for pi+ on a SPECIFIC nucleon type (sum over out-pions) -------------
_Wt, _amps = load_anl(0, 0)
def _scat_sigma_pip(W, nuc):                                  # nuc 'p' or 'n'
    tot = np.zeros_like(np.atleast_1d(W), float)
    for (pout, nout, cg) in _CHANNELS[(0, nuc)]:
        tot += np.clip(np.interp(W, _Wt, _channel_sigma(_amps, _Wt, cg), left=0, right=0), 0, None)
    return tot

def _sample_fermi(r, n):
    kf = kf_local(r)
    dirs = rng.normal(size=(n, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pmag = kf * rng.uniform(size=n) ** (1/3)
    p3 = dirs * pmag[:, None]
    E = np.sqrt(M_N**2 + pmag**2)
    return np.concatenate([E[:, None], p3], axis=1)

def _xsecs(p_pi4, p_N4, nuc):
    """abs + scatter [mb] for one pion 4-vec vs one nucleon 4-vec (arrays, leading axis = event)."""
    pE = p_pi4[:, 0]; pmom = np.linalg.norm(p_pi4[:, 1:], axis=1)
    r = None
    P = p_pi4 + p_N4
    W = np.sqrt(np.clip(P[:, 0]**2 - np.sum(P[:, 1:]**2, axis=1), 1.0, None))
    return W, pE, pmom

def two_body(p_pi4, p_N4, m_out):
    P = p_pi4 + p_N4
    s = P[:, 0]**2 - np.sum(P[:, 1:]**2, axis=1)
    sqrts = np.sqrt(np.clip(s, 1e-6, None))
    m1, m2 = m_out, M_N
    E1 = (sqrts/2) * (1 + (m1**2 - m2**2)/s)
    lam = np.sqrt(np.clip((s - m1**2 - m2**2)**2 - 4*m1**2*m2**2, 0, None))
    pf = lam/(2*sqrts)
    cth = 2*rng.uniform(size=len(s)) - 1; sth = np.sqrt(np.clip(1-cth**2, 0, None))
    phi = 2*np.pi*rng.uniform(size=len(s))
    p1cm = np.stack([E1, pf*sth*np.cos(phi), pf*sth*np.sin(phi), pf*cth], axis=1)
    beta = P[:, 1:] / P[:, 0:1]
    return _boost(p1cm, beta), P

def _boost(p4, beta):
    b2 = np.sum(beta**2, axis=1)
    g = 1/np.sqrt(np.clip(1-b2, 1e-12, None))
    bp = np.sum(beta*p4[:, 1:], axis=1)
    E = g*(p4[:, 0] + bp)
    fac = (g-1)*np.where(b2 > 1e-12, bp/np.clip(b2, 1e-12, None), 0) + g*p4[:, 0]
    vec = p4[:, 1:] + fac[:, None]*beta
    return np.concatenate([E[:, None], vec], axis=1)


# =========================================================================================== #
def run(p_lab, n_event=4000, step=0.05, mode="discrete"):
    """Shoot pi+ at carbon with impact parameter uniform in a disk of radius b_max; return the
    absorption cross section [mb].  mode='discrete' (A nucleons, impact-param walk) or
    'continuum' (mean-free-path rho*sigma along the trajectory)."""
    b_max = R_NUC                                            # incident impact-parameter disk
    A, Z = 12, 6
    m_pi = M_PIP
    Epi = np.sqrt(m_pi**2 + p_lab**2)

    # incident pions: impact parameter uniform in the disk, travelling +z from z=-(R+1)
    b = b_max * np.sqrt(rng.uniform(size=n_event))
    ang = 2*np.pi*rng.uniform(size=n_event)
    pos = np.stack([b*np.cos(ang), b*np.sin(ang), np.full(n_event, -(R_NUC+1.0))], axis=1)
    d_hat = np.tile(np.array([0., 0., 1.]), (n_event, 1))
    p_pi = np.tile(np.array([Epi, 0, 0, p_lab]), (n_event, 1)).astype(float)
    ch = np.zeros(n_event, int)                              # 0 = pi+
    M_CH = np.array([M_PIP, ox.M_PI0, M_PIP])
    alive = np.ones(n_event, bool); absorbed = np.zeros(n_event, bool)

    if mode == "discrete":
        # sample A nucleon positions per event from rho(r); first Z are protons
        # radial pdf ~ rho(r) r^2
        rr = np.linspace(0, R_NUC, 800); pdf = rho_proton(rr)*rr**2; pdf /= pdf.sum()
        rad = np.random.default_rng(2).choice(rr, size=(n_event, A), p=pdf)
        u = rng.normal(size=(n_event, A, 3)); u /= np.linalg.norm(u, axis=2, keepdims=True)
        npos = u * rad[:, :, None]                           # (n_event, A, 3)
        is_proton = np.zeros((n_event, A), bool); is_proton[:, :Z] = True
        nused = np.zeros((n_event, A), bool)
        kf_n = kf_local(rad)                                 # (n_event, A)
        # frozen Fermi momenta for the nucleons
        pmag = kf_n * rng.uniform(size=(n_event, A))**(1/3)
        ud = rng.normal(size=(n_event, A, 3)); ud /= np.linalg.norm(ud, axis=2, keepdims=True)
        nmom3 = ud * pmag[:, :, None]
        nE = np.sqrt(M_N**2 + pmag**2)
        nmom4 = np.concatenate([nE[:, :, None], nmom3], axis=2)   # (n_event, A, 4)

        nsteps = int(2.5 * (R_NUC+1) / step)
        for _ in range(nsteps):
            r = np.linalg.norm(pos, axis=1)
            esc = alive & (r > R_NUC) & (np.sum(pos*d_hat, axis=1) > 0)
            alive &= ~esc
            if not alive.any(): break
            nxt = pos + step*d_hat
            # parallel coordinate of each nucleon along d, measured from pos
            rel = npos - pos[:, None, :]                     # (E,A,3)
            par = np.sum(rel * d_hat[:, None, :], axis=2)    # (E,A)
            perp2 = np.sum(rel**2, axis=2) - par**2          # (E,A) impact param^2
            in_slab = (par > 0) & (par <= step) & (~nused) & alive[:, None]
            # cross section per (event, nucleon)
            Pp = p_pi[:, None, :] + nmom4                    # (E,A,4)
            W = np.sqrt(np.clip(Pp[:, :, 0]**2 - np.sum(Pp[:, :, 1:]**2, axis=2), 1.0, None))
            pE = p_pi[:, 0]; pmom = np.linalg.norm(p_pi[:, 1:], axis=1)
            vpi = p_pi[:, 1:]/pE[:, None]
            vN = nmom4[:, :, 1:]/nmom4[:, :, 0:1]
            vrel = np.clip(np.linalg.norm(vpi[:, None, :] - vN, axis=2), 1e-3, None)  # (E,A)
            rad_n = np.linalg.norm(npos, axis=2)
            rho_tot = 2*rho_proton(rad_n)
            kf_here = kf_local(rad_n)
            sa = np.array(ox.abs_cross_section(pE[:, None]+0*W, M_CH[ch][:, None]+0*W, pmom[:, None]+0*W,
                                               vrel, np.clip(kf_here, 1e-6, None),
                                               np.clip(rho_tot, 1e-9, None)))
            # scatter sigma depends on nucleon type & pion charge; do pi+ only here (ch stays small)
            ssc = np.where(is_proton, _scat_sigma_pip(W, "p"), _scat_sigma_pip(W, "n"))
            sa = np.clip(sa, 0, None); ssc = np.clip(ssc, 0, None)
            sig_tot = sa + ssc                               # mb
            prob = np.where(in_slab, np.exp(-np.pi*perp2/np.clip(sig_tot*MB_TO_FM2, 1e-9, None)), 0.0)
            # take the smallest-perp nucleon that passes its random test (ACHILLES: sorted, first hit)
            u_rand = rng.uniform(size=prob.shape)
            passes = in_slab & (u_rand < prob)
            big = np.where(passes, perp2, np.inf)
            hit_idx = np.argmin(big, axis=1)
            has_hit = np.isfinite(big[np.arange(n_event), hit_idx]) & alive
            # resolve interactions
            ev = np.where(has_hit)[0]
            if ev.size:
                j = hit_idx[ev]
                sa_h = sa[ev, j]; st_h = sig_tot[ev, j]
                p_abs = sa_h/np.clip(st_h, 1e-12, None)
                is_abs = rng.uniform(size=ev.size) < p_abs
                # absorption
                aev = ev[is_abs]
                absorbed[aev] = True; alive[aev] = False; nused[aev, j[is_abs]] = True
                # scatter: two-body, Pauli block on recoil -> reject (ACHILLES) if blocked
                sev = ev[~is_abs]
                if sev.size:
                    jj = j[~is_abs]
                    p_out, P = two_body(p_pi[sev], nmom4[sev, jj], M_CH[ch[sev]])
                    p_rec = P - p_out
                    kf_rec = kf_local(np.linalg.norm(npos[sev, jj], axis=1))
                    blocked = np.linalg.norm(p_rec[:, 1:], axis=1) < kf_rec
                    ok = sev[~blocked]; jok = jj[~blocked]
                    p_pi[ok] = p_out[~blocked]
                    d_hat[ok] = p_pi[ok, 1:]/np.linalg.norm(p_pi[ok, 1:], axis=1, keepdims=True)
                    nused[ok, jok] = True
            pos = pos + step*d_hat * alive[:, None]
        return np.pi * b_max**2 * absorbed.mean() * 10.0     # fm^2 -> mb (1 fm^2 = 10 mb)

    else:  # continuum mean-free-path along the trajectory (the cascade_real estimator)
        nsteps = int(2.5*(R_NUC+1)/step)
        for _ in range(nsteps):
            r = np.linalg.norm(pos, axis=1)
            esc = alive & (r > R_NUC) & (np.sum(pos*d_hat, axis=1) > 0)
            alive &= ~esc
            if not alive.any(): break
            rho_tot = 2*rho_proton(r); kf = kf_local(r)
            pN = _sample_fermi(r, n_event)
            pE = p_pi[:, 0]; pmom = np.linalg.norm(p_pi[:, 1:], axis=1)
            vpi = p_pi[:, 1:]/pE[:, None]; vN = pN[:, 1:]/pN[:, 0:1]
            vrel = np.clip(np.linalg.norm(vpi-vN, axis=1), 1e-3, None)
            P = p_pi+pN; W = np.sqrt(np.clip(P[:, 0]**2-np.sum(P[:, 1:]**2, axis=1), 1.0, None))
            sa = np.clip(np.array(ox.abs_cross_section(pE, M_CH[ch], pmom, vrel,
                            np.clip(kf, 1e-6, None), np.clip(rho_tot, 1e-9, None))), 0, None)
            # isospin-averaged scatter (p/n)
            ssc = 0.5*(_scat_sigma_pip(W, "p")+_scat_sigma_pip(W, "n"))
            sig_tot = (sa+ssc)
            lam = rho_tot*sig_tot*MB_TO_FM2
            p_int = -np.expm1(-lam*step)
            inter = alive & (rng.uniform(size=n_event) < p_int)
            p_abs = sa/np.clip(sa+ssc, 1e-12, None)
            is_abs = inter & (rng.uniform(size=n_event) < p_abs)
            absorbed |= is_abs; alive &= ~is_abs
            # scatter -> redirect (Pauli block on recoil)
            sc = inter & ~is_abs
            if sc.any():
                p_out, P2 = two_body(p_pi, pN, M_CH[ch])
                p_rec = P2 - p_out
                blk = np.linalg.norm(p_rec[:, 1:], axis=1) < kf
                do = sc & ~blk
                p_pi = np.where(do[:, None], p_out, p_pi)
                d_hat = np.where(do[:, None], p_pi[:, 1:]/np.linalg.norm(p_pi[:, 1:], axis=1, keepdims=True), d_hat)
            pos = pos + step*d_hat*alive[:, None]
        return np.pi*b_max**2*absorbed.mean()*10.0


if __name__ == "__main__":
    print(f"R_NUC = {R_NUC:.2f} fm,  disk area = {np.pi*R_NUC**2:.1f} fm^2 = {np.pi*R_NUC**2*10:.0f} mb")
    print(f"{'p_lab':>6} {'continuum':>11} {'discrete':>10}   (mb)   [DUET peak ~ 60-70 mb near 170 MeV]")
    for p in [120., 170., 220., 280., 350.]:
        sc = run(p, mode="continuum"); sd = run(p, mode="discrete")
        print(f"{p:6.0f} {sc:11.1f} {sd:10.1f}")
