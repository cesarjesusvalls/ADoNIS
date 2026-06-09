"""Test DIVERGENCE 1 (docs/res_chain_full_trace.md): ACHILLES builds ONE shared phase space
from process[0] = (n->p pi0) and evaluates ALL three CC-1pi channels' amps2 at those SHARED
(m_p, m_pi0) momenta, summing per point (Process.cc:287-302, 323-327, 357; dump-verified).

This script mirrors that: per random draw, generate ONE mu+N+pi sample with (m_Nf=M_P,
m_pi=M_PI0); then for each of the 3 channels evaluate amps2 at the SAME momenta with that
channel's (itiz, ppid) and SUM.  Compare sigma to the per-channel-mass res_xsec (0.736x) and
to ACHILLES 1.6947e-5.  If sigma -> ~1.69e-5, the shared-mass convention is the deficit cause.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
from adonis.xsec import res_xsec as R
from adonis.xsec.backend import flux_factor, MASS_PDG_PROTON, MASS_PDG_NEUTRON
from adonis.xsec.dcc_current import exclusive_amps2_batch

# process[0] = n -> p pi0 : the SHARED phase-space masses ACHILLES uses for all channels
M_P, M_PI0 = R.M_P, R.M_PI0
# the three channels (itiz, ppid, init_pid, struck_mass_for_flux) evaluated at shared momenta
CH = [(-1, 211, 2112, MASS_PDG_NEUTRON),   # n -> n pi+
      (-1, 111, 2112, MASS_PDG_NEUTRON),   # n -> p pi0
      (+1, 211, 2212, MASS_PDG_PROTON)]    # p -> p pi+


def sigma_shared(n, seed, per_channel_mass=False):
    rng = np.random.default_rng(seed)
    flux = R.T2KFlux(); maxE = flux.max_energy; minE = flux.seed_min_GeV()
    sig = 0.0
    parts = {}
    if per_channel_mass:
        # reproduce res_xsec.generate() exactly (per-channel masses) for the control number
        r = R.generate(n, seed=seed)
        return r["sigma"], {k: v for k, v in r.items() if k != "sigma"}
    # SHARED sample with process[0]=(M_P, M_PI0); ONE draw, all channels share momenta
    s = R._sample_channel(n, rng, flux, minE, maxE, M_PI0, M_P)
    v = s["valid"] & (s["energy"] > 2.5) & (s["energy"] < 400) & (s["J"] > 0)
    idx = np.where(v)[0]
    for (itiz, ppid, ipid, mstr) in CH:
        a2 = np.zeros(n)
        if len(idx):
            a2[idx] = exclusive_amps2_batch(s["k_nu"][idx], s["k_mu"][idx], s["p_struck"][idx],
                                            s["p_N"][idx], s["p_pi"][idx], itiz, ppid)
        fl = np.asarray(flux_factor(s["k_nu"], s["p_struck"], had_mass=mstr))
        w = np.where(v, a2 * fl * R.N_NUC * R.SPIN_AVG * s["J"], 0.0)
        w = np.where(np.isfinite(w) & (a2 > 0), w, 0.0)
        sc = w.mean(); sig += sc; parts[(ipid, ppid)] = sc
    return sig, parts


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ACH = 1.6947e-5
    shared = np.array([sigma_shared(n, sd)[0] for sd in range(nseed)])
    perch = np.array([sigma_shared(n, sd, per_channel_mass=True)[0] for sd in range(nseed)])
    print("ACHILLES target sigma_RES = %.4e nb" % ACH)
    print("per-channel-mass (res_xsec)  = %.4e  sem %.2e  ratio %.3f"
          % (perch.mean(), perch.std()/np.sqrt(nseed), perch.mean()/ACH))
    print("SHARED (m_p,m_pi0) all chans = %.4e  sem %.2e  ratio %.3f"
          % (shared.mean(), shared.std()/np.sqrt(nseed), shared.mean()/ACH))
    _, parts = sigma_shared(n, 0)
    print("  shared per-channel breakdown:")
    for k, val in parts.items():
        print("    init=%d pi=%d : %.4e" % (k[0], k[1], val))
