"""Free-proton mono (1 GeV) RES generator that MIRRORS ACHILLES's own event proposal exactly:
ThreeBodyMapper::GeneratePoint (TChannelMomenta for the pion split + Isotropic2Momenta for the
mu/N split) with the ACHILLES ThreeBodyMapper constants (m_alpha=0.9, ct in [-1,1], amct=1), and
weights each event by 1/gw with gw = ThreeBodyMapper::GenerateWeight (the t-channel density).

This is IDENTICAL to free_proton_gen.py in every other respect (masses, flux, spinavg, amps2,
Q2/W binning, oracle).  The ONLY change is the proposal+weight: isotropic -> ACHILLES t-channel.
=> If the dsigma/dQ2 sag closes, the bug was my independent isotropic sampler.  If it sags the
   same, the sampler is exonerated and the amps2 differs off the proposal support.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.xsec.res_xsec import M_MU, M_PIP, M_P
from adonis.xsec.dcc_current import exclusive_amps2_batch
from adonis.xsec.backend import flux_factor, MASS_PDG_PROTON
from scripts.free_proton_gen import parse_oracle, shp, chi2

E_NU = 1000.0
M_STRUCK = 939.57
# ThreeBodyMapper constexpr members (FinalStateMapper.hh)
M_ALPHA, M_CTMAX, M_CTMIN, M_AMCT = 0.9, 1.0, -1.0, 1.0
TWO_PI = 2 * np.pi


def _m2(p):
    return p[:, 0] ** 2 - (p[:, 1:] ** 2).sum(1)


def _sqlam(s, s1, s2):
    arg = (s - s1 - s2) ** 2 - 4 * s1 * s2
    return np.where(arg > 0, np.sqrt(np.clip(arg, 0, None)) / s, 0.0)


def _boost(lflag, q, ph):
    """Port of ThreeBodyMapper::Boost.  lflag=0: rest-frame-of-q -> lab.  lflag=1: lab -> rest-of-q."""
    rsq = np.sqrt(np.clip(_m2(q), 1e-12, None))
    qv, phv = q[:, 1:], ph[:, 1:]
    dot = (qv * phv).sum(1)
    if lflag == 0:
        p0 = (q[:, 0] * ph[:, 0] + dot) / rsq
        c1 = (ph[:, 0] + p0) / (rsq + q[:, 0])
        pv = phv + c1[:, None] * qv
    else:
        p0 = (q[:, 0] * ph[:, 0] - dot) / rsq
        c1 = (p0 + ph[:, 0]) / (rsq + q[:, 0])
        pv = phv - c1[:, None] * qv
    return np.concatenate([p0[:, None], pv], axis=1)


def _tj1(cn, amcxm, amcxp, ran):
    """Port of ThreeBodyMapper::Tj1 (ctexp=0.9 -> ce=0.1 branch)."""
    ce = 1.0 - cn
    return (ran * amcxm ** ce + (1.0 - ran) * amcxp ** ce) ** (1.0 / ce)


def _hj1(cn, amcxm, amcxp):
    ce = 1.0 - cn
    return (amcxp ** ce - amcxm ** ce) / ce


def _basis_from(nhat):
    """orthonormal basis with 3rd axis = nhat (Nx3)."""
    n = nhat / np.linalg.norm(nhat, axis=1, keepdims=True)
    ref = np.tile([1.0, 0.0, 0.0], (len(n), 1))
    alt = np.tile([0.0, 1.0, 0.0], (len(n), 1))
    use_alt = np.abs((n * ref).sum(1)) > 0.9
    ref = np.where(use_alt[:, None], alt, ref)
    e1 = ref - (ref * n).sum(1)[:, None] * n
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(n, e1)
    return e1, e2, n


def generate(n, m_pi=M_PIP, m_Nf=M_P, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.random((n, 5))                                  # rans[0..4]
    s2, s3, s4 = M_MU ** 2, m_Nf ** 2, m_pi ** 2
    knu = np.tile([E_NU, 0, 0, E_NU], (n, 1)).astype(float)
    pst = np.tile([M_STRUCK, 0, 0, 0], (n, 1)).astype(float)
    pin = knu + pst
    s = _m2(pin); sabs = np.sqrt(np.clip(s, 1e-12, None))
    # --- s23 uniform (mom[2]+mom[3] = mu+N system) ---
    s23max = (sabs - np.sqrt(s4)) ** 2
    s23min = np.maximum((np.sqrt(s2) + np.sqrt(s3)) ** 2, 1e-8)
    s23 = s23min + (s23max - s23min) * u[:, 0]
    # === TChannelMomenta: split off the pion (p2out, mass^2=s4); p1out=(muN), mass^2=s23 ===
    s1in = _m2(knu); s2in = _m2(pst)             # nu massless, struck^2
    s1out, s2out = s23, s4
    p1inhE = (s + s1in - s2in) / (2 * sabs)
    p1inmass = sabs * _sqlam(s, s1in, s2in) / 2
    p1outhE = (s + s1out - s2out) / (2 * sabs)
    p1outmass = sabs * _sqlam(s, s1out, s2out) / 2
    a = (0.0 - s1in - s1out + 2 * p1outhE * p1inhE) / (2 * p1inmass * p1outmass)
    a = np.where(a <= 1.0 + 1e-6, 1.0 + 1e-6, a)
    a = np.where(a < M_AMCT, M_AMCT, a)
    a = np.where(np.abs(a - M_CTMAX) < 1e-14, M_CTMAX, a)
    aminct = _tj1(M_ALPHA, a - M_CTMIN, a - M_CTMAX, u[:, 1])
    ct = a - aminct
    st = np.sqrt(np.clip(1 - ct ** 2, 0, None))
    phi = TWO_PI * u[:, 2]
    # axis = direction of p1in (nu) boosted to CM of pin
    nu_cm = _boost(1, pin, knu)
    e1, e2, nhat = _basis_from(nu_cm[:, 1:])
    dirv = (st * np.cos(phi))[:, None] * e1 + (st * np.sin(phi))[:, None] * e2 + ct[:, None] * nhat
    p1out_cm = np.concatenate([p1outhE[:, None], p1outmass[:, None] * dirv], axis=1)
    p23 = _boost(0, pin, p1out_cm)               # (muN) system in lab
    p_pi = pin - p23
    # === Isotropic2Momenta: split p23 -> mu (s2) + N (s3) ===
    rs = np.sqrt(np.clip(_m2(p23), 1e-12, None))
    p1hE = (_m2(p23) + s2 - s3) / (2 * rs)
    p1m = rs * _sqlam(_m2(p23), s2, s3) / 2
    ctB = M_CTMIN + (M_CTMAX - M_CTMIN) * u[:, 3]
    stB = np.sqrt(np.clip(1 - ctB ** 2, 0, None))
    phiB = TWO_PI * u[:, 4]
    p1h = np.stack([p1hE, p1m * stB * np.sin(phiB), p1m * stB * np.cos(phiB), p1m * ctB], axis=1)
    k_mu = _boost(0, p23, p1h)
    p_N = p23 - k_mu
    # === gw = ThreeBodyMapper::GenerateWeight (t-channel density), computed from sampled kin ===
    tcw = 2.0 * sabs / (-(a - ct) ** M_ALPHA * _hj1(M_ALPHA, a - M_CTMIN, a - M_CTMAX) * p1outmass * np.pi)
    massfac = _sqlam(_m2(p23), s2, s3)
    i2w = np.where(massfac > 1e-12, 2.0 / np.pi / np.clip(massfac, 1e-30, None) * 2.0 / (M_CTMAX - M_CTMIN), 0.0)
    gw = (2 * np.pi) ** 5 * tcw * i2w / (s23max - s23min)
    valid = (s23max > s23min) & (massfac > 1e-12) & (p1outmass > 0) & np.isfinite(gw) & (gw > 0)
    Jthree = np.where(valid, 1.0 / np.clip(gw, 1e-300, None), 0.0)
    # === amps2 (identical call to free_proton_gen) ===
    if _PS_ONLY:
        a2 = np.ones(n)
    else:
        a2 = np.zeros(n); ch = 2000
        for i in range(0, n, ch):
            sl = slice(i, min(i + ch, n))
            a2[sl] = np.asarray(exclusive_amps2_batch(knu[sl], k_mu[sl], pst[sl], p_N[sl], p_pi[sl], +1, 211))
    fl = np.asarray(flux_factor(knu, pst, had_mass=MASS_PDG_PROTON))
    w = np.where(valid, a2 * fl * 0.5 * Jthree, 0.0)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    q = knu - k_mu
    Q2 = ((q[:, 1:] ** 2).sum(1) - q[:, 0] ** 2) / 1e6
    pcm = p_N + p_pi
    W = np.sqrt(np.clip(pcm[:, 0] ** 2 - (pcm[:, 1:] ** 2).sum(1), 0, None))
    if _RETURN_MOM:
        return W, Q2, w, dict(knu=knu, k_mu=k_mu, pst=pst, p_N=p_N, p_pi=p_pi, p23=p23,
                              gw=gw, valid=valid)
    return W, Q2, w


_RETURN_MOM = False
_PS_ONLY = False


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
    print(f"generating {n} ACHILLES-mirror free-proton mono events ...", flush=True)
    W, Q2, w = generate(n)
    print(f"  kept {np.sum(w>0)}  sigma~{w.sum()/n:.4e}", flush=True)
    aW, aQ2 = parse_oracle()
    print(f"  ACHILLES oracle events: {len(aW)}", flush=True)
    We = np.linspace(1080, 1500, 26); Qe = np.linspace(0, 1.2e6, 26)
    cW, hmW, emW = shp(W, w, We); _, hoW, eoW = shp(aW, np.ones(len(aW)), We)
    cQ, hmQ, emQ = shp(Q2 * 1e6, w, Qe); _, hoQ, eoQ = shp(aQ2 * 1e6, np.ones(len(aQ2)), Qe)
    rW, srW, mW, c2W, nW = chi2(hmW, emW, hoW, eoW)
    rQ, srQ, mQ, c2Q, nQ = chi2(hmQ, emQ, hoQ, eoQ)
    cQg = cQ / 1e6
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), height_ratios=[3, 1])
    ax[0, 0].step(cW, hoW, where="mid", color="0.4", label="ACHILLES")
    ax[0, 0].errorbar(cW, hmW, emW, fmt="o-", color="C0", ms=3, label="ADoNIS (mirror)")
    ax[0, 0].legend(); ax[0, 0].set_title("dσ/dW shape  (mirror sampler, mono 1 GeV)")
    ax[1, 0].axhspan(0.97, 1.03, color="g", alpha=0.12); ax[1, 0].axhline(1, ls="--", color="g")
    ax[1, 0].errorbar(cW[mW], rW, srW, fmt="o", color="C0", ms=3); ax[1, 0].set_ylim(0.5, 1.5)
    ax[1, 0].set_xlabel("W [MeV]"); ax[1, 0].set_ylabel("ADoNIS/ACH")
    ax[1, 0].text(0.05, 0.85, f"χ²/ndf={c2W/max(nW,1):.2f}", transform=ax[1, 0].transAxes)
    ax[0, 1].step(cQg, hoQ, where="mid", color="0.4", label="ACHILLES")
    ax[0, 1].errorbar(cQg, hmQ, emQ, fmt="o-", color="C0", ms=3, label="ADoNIS (mirror)")
    ax[0, 1].legend(); ax[0, 1].set_title("dσ/dQ² shape")
    ax[1, 1].axhspan(0.97, 1.03, color="g", alpha=0.12); ax[1, 1].axhline(1, ls="--", color="g")
    ax[1, 1].errorbar(cQg[mQ], rQ, srQ, fmt="o", color="C0", ms=3); ax[1, 1].set_ylim(0.5, 1.5)
    ax[1, 1].set_xlabel("Q² [GeV²]"); ax[1, 1].set_ylabel("ADoNIS/ACH")
    ax[1, 1].text(0.05, 0.85, f"χ²/ndf={c2Q/max(nQ,1):.2f}", transform=ax[1, 1].transAxes)
    fig.suptitle(f"ACHILLES-mirror sampler vs ACHILLES   χ²/ndf W={c2W/max(nW,1):.2f} Q²={c2Q/max(nQ,1):.2f}")
    fig.tight_layout(); fig.savefig("figures/achilles_mirror_WQ.png", dpi=110)
    print(f"  chi2/ndf  W={c2W/max(nW,1):.2f}  Q2={c2Q/max(nQ,1):.2f}", flush=True)
    print(f"  Q2 ratio low(<0.2)={np.nanmean(rQ[cQg[mQ]<0.2]):.3f}  high(>0.6)={np.nanmean(rQ[cQg[mQ]>0.6]):.3f}", flush=True)
    print("  wrote figures/achilles_mirror_WQ.png")


if __name__ == "__main__":
    main()
