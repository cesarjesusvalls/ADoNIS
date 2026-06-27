"""Paper Fig. 2 (free-nucleon nu pion production) as ADoNIS-vs-ACHILLES: CC single-pion sigma(E_nu) on a
free nucleon for the three channels (p->p pi+, n->n pi+, n->p pi0), + ANL/BNL data overlay + ADO/ACH ratio.

ADoNIS side = the FAITHFUL RES cross section, NOT a toy: it reuses res_xsec's own primitives -- the 3-body
sampler `_sample_3body_dispatch`, `exclusive_amps2_batch`, `flux_factor`, `SPIN_AVG`, and the kinematic
pion mass `_pi_kin_mass` (exactly what the T2K generator integrates) -- evaluated on a STATIONARY free
nucleon at a grid of MONOCHROMATIC beam energies (the flux fold removed -> sigma(E)).

ACHILLES side = configs/achilles/run_freenucleon_res_{H,N}.yml run as an energy scan
  (python -m analysis.utils.run_achilles configs/achilles/run_freenucleon_res_H.yml --scan-energy <list>);
each mono run's GenCrossSection is sigma(E).  The N (free-neutron) run is split into n pi+ / n p pi0 by
the final-state pion charge.  Data: data/experiment/anl_bnl_cc1pi/{anl,bnl}_<chan>.txt.

Usage:  python -m analysis.paper.fig2_pion_production.make
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec.res_xsec import _sample_3body_dispatch, _pi_kin_mass, M_PIP, M_PI0, M_P, M_N, M_MU, SPIN_AVG
from adonis.xsec.dcc_current import exclusive_amps2_batch
from adonis.xsec.backend import flux_factor, MASS_PDG_PROTON, MASS_PDG_NEUTRON
from analysis.utils.hepmc import parse_events, hepmc_norm

ACH_DIR = ROOT / "output" / "achilles"
DATA = ROOT / "data" / "experiment" / "anl_bnl_cc1pi"
OUT = ROOT / "output" / "figures"
NB_TO_1E38CM2 = 1e5                      # 1 nb = 1e-33 cm^2 = 1e5 * 1e-38 cm^2

# (key, label, struck-nucleon mass, itiz, pi pid, physical pi mass, outgoing-N mass, ACH card, data stem)
CHANNELS = [
    ("ppip",  r"$p\to p\,\pi^+$",  M_P, +1, 211, M_PIP, M_P, "H", "p_ppip"),
    ("npip",  r"$n\to n\,\pi^+$",  M_N, -1, 211, M_PIP, M_N, "N", "n_npip"),
    ("nppi0", r"$n\to p\,\pi^0$",  M_N, -1, 111, M_PI0, M_P, "N", "n_ppi0"),
]
ENERGIES = np.array([0.5, 0.7, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0]) * 1000.0   # MeV


def adonis_sigma(E, m_init, itiz, pid, m_pi_phys, m_outN, n=100_000, seed=0):
    """Faithful RES sigma [nb] for one channel at a fixed (mono) neutrino energy E [MeV] on a stationary
    nucleon -- reuses the res_xsec 3-body sampler + exclusive_amps2_batch + flux_factor (no flux fold)."""
    m_pi = _pi_kin_mass(m_pi_phys); m_Nf = m_outN
    s = m_init ** 2 + 2.0 * E * m_init                       # (k_nu + p_struck)^2 with stationary nucleon
    if s < (M_MU + m_Nf + m_pi) ** 2:
        return 0.0
    rng = np.random.default_rng(seed)
    knu = np.tile([E, 0, 0, E], (n, 1)).astype(float)
    pstr = np.tile([m_init, 0, 0, 0], (n, 1)).astype(float)
    tb = _sample_3body_dispatch(knu, pstr, m_pi, m_Nf, rng.random((n, 5)))
    kmu, pN, ppi, J3, valid = tb["k_mu"], tb["p_N"], tb["p_pi"], tb["J_3body"], tb["valid3"]
    a2 = np.zeros(n); idx = np.where(np.asarray(valid) & (np.asarray(J3) > 0))[0]
    for i in range(0, len(idx), 2000):
        sl = idx[i:i + 2000]
        a2[sl] = np.asarray(exclusive_amps2_batch(knu[sl], kmu[sl], pstr[sl], pN[sl], ppi[sl], itiz, pid))
    fl = np.asarray(flux_factor(knu, pstr, had_mass=(MASS_PDG_PROTON if m_init == M_P else MASS_PDG_NEUTRON)))
    w = np.where(np.asarray(valid) & (np.asarray(J3) > 0), a2 * fl * SPIN_AVG * np.asarray(J3), 0.0)
    return float(np.nansum(w) / n)                            # nb


def achilles_sigma(card, pid=None):
    """sigma(E) [nb] from the mono energy-scan hepmc of run_freenucleon_res_<card>.yml.  If pid is given,
    return the per-final-state-pion fraction of sigma (split the free-neutron run into n pi+ / p pi0)."""
    out = {}
    for hp in sorted(ACH_DIR.glob(f"freenucleon_res_{card}_energy*.hepmc")):
        E = float(hp.stem.split("energy")[1])
        nrm = hepmc_norm(hp); sig = nrm["gen_xs_pb"] * 1e-3                  # pb -> nb (total of the run)
        if pid is not None:                                                  # split by final-state pion charge
            npi = nsel = 0
            for ev in parse_events(hp):
                pis = [p for p, st, _ in ev["parts"] if p in (211, 111, -211) and st == 1]
                if not pis:
                    continue
                npi += 1; nsel += int(pis[0] == pid)
            sig *= (nsel / npi) if npi else 0.0
        out[E] = sig
    return out


def main(argv=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), height_ratios=[3, 1], sharex="col")
    any_ach = False
    cache = OUT / "fig2_adonis_sigma.npz"                              # persist ADoNIS sigma (re-render w/o recompute)
    cached = dict(np.load(cache)) if (cache.exists() and "--recompute" not in (argv or sys.argv[1:])) else {}
    new_cache = {}
    for c, (key, label, m_i, itiz, pid, mpi, mNf, card, stem) in enumerate(CHANNELS):
        a0, a1 = axes[0, c], axes[1, c]
        if key in cached:
            ado = cached[key]
        else:
            ado = np.array([adonis_sigma(E, m_i, itiz, pid, mpi, mNf) for E in ENERGIES]) * NB_TO_1E38CM2
        new_cache[key] = ado
        a0.plot(ENERGIES / 1000, ado, "s-", color="C0", ms=4, lw=1.0, label="ADoNIS")
        ach = achilles_sigma(card, pid if card == "N" else None)
        if ach:
            any_ach = True
            xe = np.array(sorted(ach)); ya = np.array([ach[e] for e in xe]) * NB_TO_1E38CM2
            a0.plot(xe / 1000, ya, "D--", color="0.35", ms=4, lw=1.0, label="ACHILLES")
            ad_i = np.interp(xe, ENERGIES, ado)
            a1.axhspan(0.9, 1.1, color="green", alpha=0.12); a1.axhline(1.0, ls="--", color="green", lw=0.7)
            with np.errstate(divide="ignore", invalid="ignore"):
                a1.plot(xe / 1000, ad_i / np.where(ya > 0, ya, np.nan), "o", color="C3", ms=4)
        for tag, mk in (("anl", "o"), ("bnl", "^")):
            f = DATA / f"{tag}_{stem}.txt"
            if f.exists():
                d = np.loadtxt(f); xd = d[:, 0]; yd = d[:, 1] / 1e-38; ye = d[:, 2] / 1e-38
                a0.errorbar(xd, yd, yerr=ye, fmt=mk, color="k", ms=4, capsize=2, lw=0.8,
                            label=f"{tag.upper()} data", zorder=5)
        a0.set_title(label, fontsize=11); a0.set_ylim(bottom=0); a0.set_xlim(0, ENERGIES.max() / 1000 + 0.3)
        a0.legend(fontsize=8)
        a0.set_ylabel(r"$\sigma$ [$10^{-38}$ cm$^2$]") if c == 0 else None
        a1.set_ylim(0.5, 1.5); a1.set_xlabel(r"$E_\nu$ [GeV]"); a1.set_ylabel("ADO/ACH") if c == 0 else None
    fig.suptitle(r"Free-nucleon CC single-pion $\sigma(E_\nu)$ — ADoNIS vs ACHILLES vs ANL/BNL (paper Fig. 2)",
                 fontsize=13)
    os.makedirs(OUT, exist_ok=True)
    np.savez(cache, **new_cache)                                       # cache ADoNIS sigma for instant re-render
    out = OUT / "fig2_pion_production.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); print("wrote", out, flush=True)
    if not any_ach:
        print("  (no ACHILLES scan yet -- run: python -m analysis.utils.run_achilles "
              "configs/achilles/run_freenucleon_res_H.yml --scan-energy 500,700,1000,1250,1500,2000,3000,4000,6000"
              " ; same for _N)", flush=True)


if __name__ == "__main__":
    main()
