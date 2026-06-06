"""Step A: faithful discrete impact-parameter cascade replicating ACHILLES Cascade.cc (Gaussian
probability over explicit nucleons), to validate it reproduces the ACHILLES pi+-12C transparency
oracle now that step B proved the per-nucleon sigma_abs is bit-exact.

ACHILLES algorithm (Evolve/Interacted): A nucleons placed from the QMC configs, each with a
local-Fermi-gas momentum; the test pion is fired at impact parameter b over the nuclear disk;
each step it finds background nucleons in the swept slab, computes sigma=sigma_scatter+sigma_abs
with that nucleon's actual v_rel, prob=exp(-pi b_perp^2/(sigma/10)), interacts with the first
(smallest b_perp) that passes, branches abs/scatter by sigma_abs/sigma_tot, CONSUMES the struck
nucleon, Pauli-blocks outgoing nucleons.  sigma_abs uses the smooth rho(r) at the nucleon radius
(as in ACHILLES Oset.AbsCrossSection).
"""
import os, sys, gzip
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import adonis.fsi.oset_xsec as ox
from adonis.fsi.mb.anl_xsec import load_anl, _channel_sigma, dsigma_dOmega
from adonis.fsi.mb.cascade_mb import _CHANNELS

# DCC angular inverse-CDF: cos_cm sampled from dsigma/dOmega (1+3cos^2 at the Delta), per W
_Wg = np.linspace(1085., 1700., 96); _cg = np.linspace(-1, 1, 181); _ug = np.linspace(0, 1, 64)
_INV = np.zeros((_Wg.size, _ug.size))
for _i, _W in enumerate(_Wg):
    _dd = np.clip(np.asarray(dsigma_dOmega(float(_W), _cg)), 0, None)
    _cd = np.concatenate([[0], np.cumsum(0.5 * (_dd[1:] + _dd[:-1]) * np.diff(_cg))])
    _cd = _cd / _cd[-1] if _cd[-1] > 0 else np.linspace(0, 1, _cg.size)
    _INV[_i] = np.interp(_ug, _cd, _cg)
def sample_cos_cm(W, u):
    iw = np.clip(((W - _Wg[0]) / (_Wg[1] - _Wg[0])).astype(int), 0, _Wg.size - 1)
    return np.array([np.interp(u[k], _ug, _INV[iw[k]]) for k in range(len(u))])

HBARC = ox.HBARC; M_N = ox.M_N; M_PIP = ox.M_PIP; MB = 0.1
rng = np.random.default_rng(0)
d = np.loadtxt("data/nuclear/c12_density.txt", comments="#"); rg, rp = d[:, 0], d[:, 1]
R_NUC = float(rg[rp > 1e-4 * rp[0]].max())
rho_p = lambda r: np.interp(r, rg, rp, left=rp[0], right=0.0)
kf_loc = lambda r: np.cbrt(np.clip(rho_p(r), 0, None) * 3 * np.pi ** 2) * HBARC

# DCC scatter sigma per target type on the W grid (sum over out-pions), for pi+
Wt, amps = load_anl(0, 0)
_sc = {nuc: sum(np.clip(_channel_sigma(amps, Wt, cg), 0, None) for _, _, cg in _CHANNELS[(0, nuc)])
       for nuc in ("p", "n")}
scat = lambda W, isp: np.where(isp, np.interp(W, Wt, _sc["p"], left=0, right=0),
                               np.interp(W, Wt, _sc["n"], left=0, right=0))

# load QMC configs (positions [fm], isospin, per-config weight)
def load_configs(nmax=20000):
    A = 12; iso = np.zeros((nmax, A), bool); pos = np.zeros((nmax, A, 3)); wt = np.zeros(nmax)
    with gzip.open("../Achilles/data/configurations/QMC_configs.out.gz", "rt") as f:
        f.readline()
        for c in range(nmax):
            for i in range(A):
                t = f.readline().split(); iso[c, i] = float(t[0]) > 0
                pos[c, i] = [float(t[1]), float(t[2]), float(t[3])]
            wt[c] = float(f.readline()); f.readline()
    return iso, pos, wt
_ISO, _POS, _WT = load_configs()
_WT = _WT / _WT.sum()


def sample_nucleons(n):
    """Pick n configs ~ weight; assign each nucleon a local-Fermi-gas momentum."""
    idx = rng.choice(len(_WT), size=n, p=_WT)
    pos = _POS[idx].copy(); isp = _ISO[idx].copy()                    # (n,12,3),(n,12)
    r = np.linalg.norm(pos, axis=2)
    kf = kf_loc(r)
    dirs = rng.normal(size=pos.shape); dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
    pm = kf * rng.uniform(size=r.shape) ** (1 / 3)
    p3 = dirs * pm[:, :, None]; E = np.sqrt(M_N ** 2 + pm ** 2)
    mom = np.concatenate([E[:, :, None], p3], axis=3) if False else np.concatenate([E[..., None], p3], axis=2)
    return pos, mom, isp


def run(plab, n=4000, step=0.05):
    A = 12; bmax = R_NUC
    b = bmax * np.sqrt(rng.uniform(size=n)); ang = 2 * np.pi * rng.uniform(size=n)
    Epi = np.sqrt(M_PIP ** 2 + plab ** 2)
    pos = np.stack([b * np.cos(ang), b * np.sin(ang),
                    -np.sqrt(np.clip(R_NUC ** 2 - b ** 2, 0, None)) * 0.999], 1)
    dhat = np.tile([0, 0, 1.], (n, 1)); p_pi = np.tile([Epi, 0, 0, plab], (n, 1)).astype(float)
    npos, nmom, nisp = sample_nucleons(n)
    consumed = np.zeros((n, A), bool); alive = np.ones(n, bool); absb = np.zeros(n, bool); react = np.zeros(n, bool)
    for _ in range(int(2.6 * (R_NUC + 1) / step)):
        moving = alive & (np.sum(pos * dhat, 1) > 0) & (np.linalg.norm(pos, axis=1) > R_NUC)
        alive &= ~moving
        if not alive.any(): break
        rel = npos - pos[:, None, :]
        par = np.sum(rel * dhat[:, None, :], axis=2)
        perp2 = np.sum(rel ** 2, axis=2) - par ** 2
        in_slab = (par > 0) & (par <= step) & (~consumed) & alive[:, None]
        # cross sections per (n,A)
        pE = p_pi[:, 0]; pmom = np.linalg.norm(p_pi[:, 1:], axis=1)
        vpi = p_pi[:, 1:] / pE[:, None]
        vN = nmom[:, :, 1:] / nmom[:, :, 0:1]
        vrel = np.clip(np.linalg.norm(vpi[:, None, :] - vN, axis=2), 1e-3, None)
        rnuc = np.linalg.norm(npos, axis=2)
        rho_t = 2 * rho_p(rnuc); kf_n = kf_loc(rnuc)
        Pp = p_pi[:, None, :] + nmom
        W = np.sqrt(np.clip(Pp[:, :, 0] ** 2 - np.sum(Pp[:, :, 1:] ** 2, axis=2), 1., None))
        sa = np.clip(np.array(ox.abs_cross_section(pE[:, None] + 0 * W, M_PIP, pmom[:, None] + 0 * W,
                     vrel, np.clip(kf_n, 1e-6, None), np.clip(rho_t, 1e-9, None))), 0, None)
        ss = np.clip(scat(W, nisp), 0, None)
        sig = sa + ss
        prob = np.where(in_slab, np.exp(-np.pi * perp2 / np.clip(sig * MB, 1e-12, None)), 0.0)
        u = rng.uniform(size=prob.shape)
        passes = in_slab & (u < prob)
        big = np.where(passes, perp2, np.inf)
        j = np.argmin(big, axis=1)
        hit = np.isfinite(big[np.arange(n), j]) & alive
        ev = np.where(hit)[0]
        if ev.size:
            jj = j[ev]
            sa_h = sa[ev, jj]; st_h = sig[ev, jj]
            p_abs = sa_h / np.clip(st_h, 1e-12, None)
            ua = rng.uniform(size=ev.size)
            is_abs = ua < p_abs
            # absorption
            aev = ev[is_abs]
            absb[aev] = True; react[aev] = True; alive[aev] = False; consumed[aev, jj[is_abs]] = True
            # scatter: 2-body isotropic-CM placeholder (DCC angular added in production); Pauli block recoil
            sev = ev[~is_abs]
            if sev.size:
                js = jj[~is_abs]
                pN_i = nmom[sev, js]; P = p_pi[sev] + pN_i
                s = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1); sq = np.sqrt(np.clip(s, 1e-6, None))
                E1 = (sq / 2) * (1 + (M_PIP ** 2 - M_N ** 2) / s)
                lam = np.sqrt(np.clip((s - M_PIP ** 2 - M_N ** 2) ** 2 - 4 * M_PIP ** 2 * M_N ** 2, 0, None))
                pf = lam / (2 * sq)
                ct = sample_cos_cm(sq, rng.uniform(size=sev.size)); st = np.sqrt(np.clip(1 - ct ** 2, 0, None))
                ph = 2 * np.pi * rng.uniform(size=sev.size)
                p1cm = np.stack([E1, pf * st * np.cos(ph), pf * st * np.sin(ph), pf * ct], 1)
                beta = P[:, 1:] / P[:, 0:1]
                # boost to lab
                b2 = np.sum(beta ** 2, axis=1); g = 1 / np.sqrt(np.clip(1 - b2, 1e-12, None))
                bp = np.sum(beta * p1cm[:, 1:], axis=1)
                Eout = g * (p1cm[:, 0] + bp)
                fac = (g - 1) * np.where(b2 > 1e-12, bp / np.clip(b2, 1e-12, None), 0) + g * p1cm[:, 0]
                vout = p1cm[:, 1:] + fac[:, None] * beta
                p_out = np.concatenate([Eout[:, None], vout], 1)
                p_rec = P - p_out
                kf_rec = kf_loc(np.linalg.norm(npos[sev, js], axis=1))
                blocked = np.linalg.norm(p_rec[:, 1:], axis=1) < kf_rec
                ok = sev[~blocked]; jok = js[~blocked]
                p_pi[ok] = p_out[~blocked]
                dhat[ok] = p_pi[ok, 1:] / np.linalg.norm(p_pi[ok, 1:], axis=1, keepdims=True)
                react[ok] = True; consumed[ok, jok] = True
        pos = pos + step * dhat * alive[:, None]
    geo = np.pi * bmax ** 2 * 10.0
    return geo * absb.mean(), geo * react.mean()


if __name__ == "__main__":
    orc = np.loadtxt("data/oracle/cascade_pip_c12_virt_abs.csv")   # correct radius=10fm norm
    print(f"{'p':>5} {'orc_abs':>8} {'disc_abs':>9} {'ratio':>6} | {'orc_reac':>9} {'disc_reac':>10} {'ratio':>6}")
    for p in [155, 215, 275, 335, 395]:
        sa, sr = run(float(p), n=8000)
        oa = np.interp(p, orc[:, 0], orc[:, 2]); o_r = np.interp(p, orc[:, 0], orc[:, 1])
        print(f"{p:5d} {oa:8.1f} {sa:9.1f} {sa/oa:6.2f} | {o_r:9.1f} {sr:10.1f} {sr/o_r:6.2f}")
