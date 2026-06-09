"""Free-proton mono (1 GeV) RES generator using the EXACT res_xsec 3-body sampler (the
bit-exact-validated psw) + the proven exclusive amps2 + flux. Single channel p->p pi+.
Histograms dsigma/dW, dsigma/dQ2 and compares to the ACHILLES oracle (nofsi_res_H_mono.hepmc),
with chi2/ndf and propagated stat errors. If all five weight factors are truly correct, this
must agree -- this is the direct test of that claim for a single proton."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.xsec.res_xsec import _sqlam, _boost_to_lab, M_MU, _TWO_PI, M_PIP, M_P
from adonis.xsec.dcc_current import exclusive_amps2_batch
from adonis.xsec.backend import flux_factor, MASS_PDG_PROTON

E_NU = 1000.0
M_STRUCK = 939.57          # ACHILLES struck-nucleon energy in the mono free-proton RESDUMP
ACHHEPMC = Path("/Users/homelab/Lab/Playground/projects/DIFFGEN/Achilles/_resrun_out/nofsi_res_H_mono.hepmc")


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.random((n, 5))                                  # s23, ctA, phA, ctB, phB
    m_pi = M_PIP; m_Nf = M_P
    knu = np.tile([E_NU, 0, 0, E_NU], (n, 1)).astype(float)
    pst = np.tile([M_STRUCK, 0, 0, 0], (n, 1)).astype(float)
    P = knu + pst
    s = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1)
    sqrts = np.sqrt(np.clip(s, 1e-9, None))
    s23max = (sqrts - m_pi) ** 2; s23min = max((M_MU + m_Nf) ** 2, 1e-8)
    s23 = s23min + (s23max - s23min) * u[:, 0]; rs23 = np.sqrt(np.clip(s23, 1e-9, None))
    # split A: total -> (muN) + pi
    EmuN = (s + s23 - m_pi ** 2) / (2 * sqrts); pA = sqrts * _sqlam(s, s23, m_pi ** 2) / 2
    ctA = 2 * u[:, 1] - 1; stA = np.sqrt(np.clip(1 - ctA ** 2, 0, None)); phA = _TWO_PI * u[:, 2]
    dA = np.stack([stA * np.cos(phA), stA * np.sin(phA), ctA], axis=1)
    muN_cm = np.concatenate([EmuN[:, None], pA[:, None] * dA], axis=1)
    pi_cm = np.concatenate([np.sqrt(m_pi ** 2 + pA ** 2)[:, None], -pA[:, None] * dA], axis=1)
    p_muN = _boost_to_lab(muN_cm, P); p_pi = _boost_to_lab(pi_cm, P)
    I2W_A = 2.0 / np.pi / np.clip(_sqlam(s, s23, m_pi ** 2), 1e-12, None)
    # split B: (muN) -> mu + N
    Emu = (s23 + M_MU ** 2 - m_Nf ** 2) / (2 * rs23); pB = rs23 * _sqlam(s23, M_MU ** 2, m_Nf ** 2) / 2
    ctB = 2 * u[:, 3] - 1; stB = np.sqrt(np.clip(1 - ctB ** 2, 0, None)); phB = _TWO_PI * u[:, 4]
    dB = np.stack([stB * np.cos(phB), stB * np.sin(phB), ctB], axis=1)
    mu_cm = np.concatenate([Emu[:, None], pB[:, None] * dB], axis=1)
    N_cm = np.concatenate([np.sqrt(m_Nf ** 2 + pB ** 2)[:, None], -pB[:, None] * dB], axis=1)
    k_mu = _boost_to_lab(mu_cm, p_muN); p_N = _boost_to_lab(N_cm, p_muN)
    I2W_B = 2.0 / np.pi / np.clip(_sqlam(s23, M_MU ** 2, m_Nf ** 2), 1e-12, None)
    density = (2 * np.pi) ** 5 * I2W_A * I2W_B / (s23max - s23min)
    J_3body = np.where(density > 0, 1.0 / np.clip(density, 1e-300, None), 0.0)
    valid = ((s23max > s23min) & (_sqlam(s, s23, m_pi ** 2) > 0) & (_sqlam(s23, M_MU ** 2, m_Nf ** 2) > 0))
    # weights: amps2 * flux * spinavg * J_3body (initwgt=1, mono beam -> no beam jac)
    a2 = np.zeros(n); ch = 1000
    for i in range(0, n, ch):
        sl = slice(i, min(i + ch, n))
        a2[sl] = np.asarray(exclusive_amps2_batch(knu[sl], k_mu[sl], pst[sl], p_N[sl], p_pi[sl], +1, 211))
    fl = np.asarray(flux_factor(knu, pst, had_mass=MASS_PDG_PROTON))
    w = np.where(valid, a2 * fl * 0.5 * J_3body, 0.0)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    q = knu - k_mu
    Q2 = ((q[:, 1:] ** 2).sum(1) - q[:, 0] ** 2) / 1e6
    pcm = p_N + p_pi
    W = np.sqrt(np.clip(pcm[:, 0] ** 2 - (pcm[:, 1:] ** 2).sum(1), 0, None))
    return W, Q2, w


def parse_oracle():
    W, Q2 = [], []
    nu, mu, pN, pPi = [], None, None, None
    def m2(p): return p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2
    def flush():
        nonlocal nu, mu, pN, pPi
        if nu and mu is not None and pN is not None and pPi is not None:
            knu = max(nu, key=lambda p: p[0]); q = knu - mu
            Q2.append(-m2(q) / 1e6); W.append(np.sqrt(max(m2(pN + pPi), 0.0)))
        nu, mu, pN, pPi = [], None, None, None
    with open(ACHHEPMC) as fh:
        for line in fh:
            t = line[:2]
            if t == "E ": flush()
            elif t == "P ":
                f = line.split(); pid = int(f[3]); st = int(f[9])
                p4 = np.array([float(f[7]), float(f[4]), float(f[5]), float(f[6])])
                if pid == 14: nu.append(p4)
                elif pid == 13 and st == 1: mu = p4
                elif st == 1 and abs(pid) in (2112, 2212): pN = p4
                elif st == 1 and pid in (211, 111, -211): pPi = p4
    flush()
    return np.array(W), np.array(Q2)


def shp(v, w, e):
    c = 0.5 * (e[1:] + e[:-1]); bw = np.diff(e)
    h, _ = np.histogram(v, bins=e, weights=w); h2, _ = np.histogram(v, bins=e, weights=w**2)
    d = h / bw; er = np.sqrt(h2) / bw
    a = np.sum(0.5 * (d[1:] + d[:-1]) * np.diff(c))
    return c, d / a, er / a


def chi2(hm, em, ho, eo):
    m = (ho > 0) & (hm > 0)
    r = hm[m] / ho[m]; sr = r * np.sqrt((em[m]/hm[m])**2 + (eo[m]/ho[m])**2)
    return r, sr, m, float(np.sum((r-1)**2/sr**2)), int(m.sum())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
    print(f"generating {n} free-proton mono events ...", flush=True)
    W, Q2, w = generate(n)
    print(f"  kept {np.sum(w>0)}  sigma~{w.sum()/n:.4e}", flush=True)
    aW, aQ2 = parse_oracle()
    print(f"  ACHILLES oracle events: {len(aW)}", flush=True)
    We = np.linspace(1080, 1500, 26); Qe = np.linspace(0, 1.2e6, 26)
    cW, hmW, emW = shp(W, w, We); _, hoW, eoW = shp(aW, np.ones(len(aW)), We)
    cQ, hmQ, emQ = shp(Q2*1e6, w, Qe); _, hoQ, eoQ = shp(aQ2*1e6, np.ones(len(aQ2)), Qe)
    rW, srW, mW, c2W, nW = chi2(hmW, emW, hoW, eoW)
    rQ, srQ, mQ, c2Q, nQ = chi2(hmQ, emQ, hoQ, eoQ)
    cQg = cQ/1e6
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), height_ratios=[3, 1])
    ax[0,0].step(cW, hoW, where="mid", color="0.4", label="ACHILLES")
    ax[0,0].errorbar(cW, hmW, emW, fmt="o-", color="C3", ms=3, label="ADoNIS (free p)")
    ax[0,0].legend(); ax[0,0].set_title("dσ/dW shape  (single proton, mono 1 GeV)")
    ax[1,0].axhspan(0.97,1.03,color="g",alpha=0.12); ax[1,0].axhline(1, ls="--", color="g")
    ax[1,0].errorbar(cW[mW], rW, srW, fmt="o", color="C3", ms=3); ax[1,0].set_ylim(0.5,1.5)
    ax[1,0].set_xlabel("W [MeV]"); ax[1,0].set_ylabel("ADoNIS/ACH")
    ax[1,0].text(0.05,0.85,f"χ²/ndf={c2W/max(nW,1):.2f}", transform=ax[1,0].transAxes)
    ax[0,1].step(cQg, hoQ, where="mid", color="0.4", label="ACHILLES")
    ax[0,1].errorbar(cQg, hmQ, emQ, fmt="o-", color="C3", ms=3, label="ADoNIS")
    ax[0,1].legend(); ax[0,1].set_title("dσ/dQ² shape")
    ax[1,1].axhspan(0.97,1.03,color="g",alpha=0.12); ax[1,1].axhline(1, ls="--", color="g")
    ax[1,1].errorbar(cQg[mQ], rQ, srQ, fmt="o", color="C3", ms=3); ax[1,1].set_ylim(0.5,1.5)
    ax[1,1].set_xlabel("Q² [GeV²]"); ax[1,1].set_ylabel("ADoNIS/ACH")
    ax[1,1].text(0.05,0.85,f"χ²/ndf={c2Q/max(nQ,1):.2f}", transform=ax[1,1].transAxes)
    fig.suptitle(f"Single proton, mono 1 GeV: ADoNIS vs ACHILLES   χ²/ndf W={c2W/max(nW,1):.2f} Q²={c2Q/max(nQ,1):.2f}")
    fig.tight_layout(); fig.savefig("figures/free_proton_WQ.png", dpi=110)
    print(f"  chi2/ndf  W={c2W/max(nW,1):.2f}  Q2={c2Q/max(nQ,1):.2f}", flush=True)
    print(f"  Q2 ratio low(<0.2)={np.nanmean(rQ[cQg[mQ]<0.2]):.3f}  high(>0.6)={np.nanmean(rQ[cQg[mQ]>0.6]):.3f}", flush=True)
    print("  wrote figures/free_proton_WQ.png")


if __name__ == "__main__":
    main()
