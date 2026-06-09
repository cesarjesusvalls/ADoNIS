"""Mono 1 GeV free-proton: compare NORMALISED distributions of ACHILLES hepmc vs ADoNIS forward
across Q2, W, cos(theta_pi in Npi-CM rel q), cos(theta_mu lab).  Find which variable diverges."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import scripts.test_3body_q2_measure as T
from adonis.xsec.dcc_current import exclusive_amps2_batch
from adonis.xsec.backend import flux_factor, MASS_PDG_PROTON
ACH = Path(__file__).resolve().parents[2] / "Achilles/_resrun_out/nofsi_res_H_mono.hepmc"


def mink2(p): return p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2
def boost_to_rest(p, P):
    M = np.sqrt(np.clip(P[:, 0]**2 - np.sum(P[:, 1:]**2, axis=1), 1e-9, None))
    E = (P[:, 0]*p[:, 0] - np.sum(P[:, 1:]*p[:, 1:], axis=1)) / M
    c = (p[:, 0] + E) / (M + P[:, 0])
    return np.concatenate([E[:, None], p[:, 1:] - c[:, None]*P[:, 1:]], axis=1)


def variables(knu, kmu, pN, ppi):
    q = knu - kmu
    Q2 = (np.sum(q[:, 1:]**2, axis=1) - q[:, 0]**2) / 1e6
    pcm = pN + ppi
    W = np.sqrt(np.clip(pcm[:, 0]**2 - np.sum(pcm[:, 1:]**2, axis=1), 0, None))
    pi_r = boost_to_rest(ppi, pcm); q_r = boost_to_rest(q, pcm)
    cpi = np.sum(pi_r[:, 1:]*q_r[:, 1:], axis=1) / (np.linalg.norm(pi_r[:, 1:], axis=1)*np.linalg.norm(q_r[:, 1:], axis=1) + 1e-9)
    cmu = kmu[:, 3] / np.linalg.norm(kmu[:, 1:], axis=1)        # muon cos relative to beam (z)
    return Q2, W, cpi, cmu


def parse_ach():
    nu, mu, pN, ppi = [], None, None, None
    out = []
    def flush():
        nonlocal nu, mu, pN, ppi
        if nu and mu is not None and pN is not None and ppi is not None:
            out.append((max(nu, key=lambda p: p[0]), mu, pN, ppi))
        nu, mu, pN, ppi = [], None, None, None
    for line in open(ACH):
        t = line[:2]
        if t == "E ":
            flush()
        elif t == "P ":
            f = line.split(); pid = int(f[3]); st = int(f[9])
            p4 = np.array([float(f[7]), float(f[4]), float(f[5]), float(f[6])])
            if pid == 14: nu.append(p4)
            elif pid == 13 and st == 1: mu = p4
            elif pid == 2212 and st == 1: pN = p4
            elif pid == 211 and st == 1: ppi = p4
    flush()
    a = np.array(out)  # (n,4,4)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def adonis(n=600000):
    rng = np.random.default_rng(0)
    knu = np.tile(T.k_nu, (n, 1)); pstk = np.tile(T.p_st, (n, 1))
    q2, j3, kmu, pN, ppi, val = T.isotropic_full(n, rng)
    a = np.zeros(n); idx = np.where(val)[0]
    a[idx] = np.asarray(exclusive_amps2_batch(knu[idx], kmu[idx], pstk[idx], pN[idx], ppi[idx], +1, 211))
    a = np.where(np.isfinite(a), a, 0)
    fl = np.asarray(flux_factor(knu, pstk, had_mass=MASS_PDG_PROTON))
    w = np.where(np.isfinite(a*fl*0.5*j3*val), a*fl*0.5*j3*val, 0)
    keep = w > 0
    return kmu[keep], pN[keep], ppi[keep], knu[keep], w[keep]


if __name__ == "__main__":
    aknu, akmu, apN, appi = parse_ach()
    print(f"ACHILLES mono events: {len(aknu)}")
    aQ2, aW, acpi, acmu = variables(aknu, akmu, apN, appi)
    aw = np.ones(len(aknu))                          # unweighted physical events
    dkmu, dpN, dppi, dknu, dw = adonis()
    dQ2, dW, dcpi, dcmu = variables(dknu, dkmu, dpN, dppi)
    print("Normalised-shape ratio ADoNIS/ACHILLES per bin (where physical event distributions should MATCH if amps2 agrees):")
    for name, av, dv, bins in [("Q2[GeV2]", aQ2, dQ2, np.linspace(0, 1.5, 11)),
                               ("W[MeV]", aW, dW, np.linspace(1080, 1700, 11)),
                               ("cos_pi(Npi-CM/q)", acpi, dcpi, np.linspace(-1, 1, 11)),
                               ("cos_mu(lab/beam)", acmu, dcmu, np.linspace(0.9, 1.0, 11))]:
        ha, _ = np.histogram(av, bins=bins, weights=aw); ha = ha/ha.sum()
        hd, _ = np.histogram(dv, bins=bins, weights=dw); hd = hd/hd.sum()
        ctr = 0.5*(bins[1:]+bins[:-1])
        print(f"\n {name}:")
        for i in range(len(ctr)):
            if ha[i] > 0:
                print(f"   {ctr[i]:8.3f}   ACH {ha[i]:.4f}  ADO {hd[i]:.4f}   ADO/ACH {hd[i]/ha[i]:.3f}")
