"""ADoNIS T2K CC0pi-Np STV prediction: flux-folded CCQE (Llewellyn-Smith) off the spectral-
function nuclear model, with proton FSI (GiBUU NN-elastic), reduced to delta_pT / delta_alphaT.

Kinematics: nu(+z) + n(p_struck from S(p,E)) -> mu + p.  The muon (E',theta) is reconstructed
from (E_nu, Q^2) via the free QE relation (nucleon mass); q = k_nu - k_mu; the outgoing proton
3-momentum = p_struck + q (on-shell).  Transverse momentum conservation makes delta_pT = the
struck-nucleon transverse momentum before FSI; proton rescattering adds the high-delta_pT tail.
Weight = LS dsigma/dQ^2 (sampled Q^2 uniform -> x proposal volume); E_nu ~ T2K flux.

T2K CC0pi-Np cuts: p_mu>250, cos_mu>-0.6, 450<p_p<1000 MeV/c, cos_p>0.4.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.flux.spectrum import Spectrum
from adonis.nuclear.spectral import SpectralFunction
from adonis.primary.qe.llewellyn_smith import ls_dsigma_dQ2
from adonis.core.event import EventRecord
from adonis.fsi.cascade_discrete import DiscreteNucleonFSI, DiscreteCascadeConfig
from adonis import observables as obs

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "_paper_release" / "AchillesGen-arXiv-2508.19213-48a9148"
M_MU = 105.6583745
M_N = 939.0          # MeV (mean nucleon)
M_P = 938.272


def generate(n=1_000_000, seed=0, with_fsi=True, MA_GeV=1.03):
    flux = Spectrum(REL / "T2K" / "T2K_nu.dat")
    nuc = SpectralFunction("pke12p_tot.data")
    key = jax.random.PRNGKey(seed)
    ke, kq, ksf, kph, knuc = jax.random.split(key, 5)
    enu = np.asarray(flux.sample_enu(ke, n))                      # MeV
    # sample Q^2 uniform in [0, Q2hi]; weight by LS dsigma/dQ^2
    Q2hi = np.minimum(2.0 * M_N * enu, 3.0e6)                     # MeV^2 cap
    Q2 = np.asarray(jax.random.uniform(kq, (n,))) * Q2hi          # MeV^2
    enu_g, Q2_g = enu / 1000.0, Q2 / 1e6                          # GeV, GeV^2
    dsig = np.asarray(ls_dsigma_dQ2(Q2_g, enu_g, MA=MA_GeV, m_l=M_MU / 1000.0))
    wq = np.clip(dsig, 0.0, None) * (Q2hi / 1e6)                  # x proposal volume [GeV^2]

    # free-QE muon reconstruction (nucleon at rest): omega = Q^2/(2M), E' = E_nu - omega
    omega = Q2 / (2.0 * M_N)
    Emu = enu - omega
    pmu = np.sqrt(np.clip(Emu ** 2 - M_MU ** 2, 0.0, None))
    cth = (2.0 * enu * Emu - Q2 - M_MU ** 2) / (2.0 * enu * np.clip(pmu, 1e-6, None))
    valid = (Emu > M_MU) & (np.abs(cth) <= 1.0) & (wq > 0)
    cth = np.clip(cth, -1.0, 1.0); sth = np.sqrt(np.clip(1 - cth ** 2, 0.0, None))
    phi = np.asarray(jax.random.uniform(kph, (n,))) * 2 * np.pi
    kmu = np.stack([Emu, pmu * sth * np.cos(phi), pmu * sth * np.sin(phi), pmu * cth], axis=1)
    knu = np.stack([enu, np.zeros(n), np.zeros(n), enu], axis=1)
    q = knu - kmu

    p_vec, E_rm = nuc.sample_nucleon(ksf, n)
    p_vec = np.asarray(p_vec)
    p_p3 = p_vec + q[:, 1:]
    Ep = np.sqrt(M_P ** 2 + np.sum(p_p3 ** 2, axis=1))
    p_p = np.concatenate([Ep[:, None], p_p3], axis=1)

    w = np.where(valid, wq, 0.0)
    z = jnp.zeros((n, 4))
    ev = EventRecord(k=jnp.asarray(knu), kp=jnp.asarray(kmu), p_struck=jnp.asarray(
        np.concatenate([(M_N - np.asarray(E_rm))[:, None], p_vec], axis=1)),
        p_pi=z, p_N=jnp.asarray(p_p), w=jnp.asarray(w),
        channel=jnp.zeros(n, jnp.int32), pid_pi=jnp.zeros(n, jnp.int32),
        pid_N=jnp.full((n,), 2212), pid_Ni=jnp.full((n,), 2112),
        W=jnp.zeros(n), Q2_adj=jnp.asarray(Q2))
    if with_fsi:
        ev = DiscreteNucleonFSI(DiscreteCascadeConfig(seed=1, step=0.05, max_steps=260)).apply(None, ev, key=knuc)

    mu = np.asarray(ev.kp); pp = np.asarray(ev.p_N); wv = np.asarray(ev.w)
    pmu_f = np.linalg.norm(mu[:, 1:], axis=1); cmu = mu[:, 3] / np.clip(pmu_f, 1e-6, None)
    ppm = np.linalg.norm(pp[:, 1:], axis=1); cp = pp[:, 3] / np.clip(ppm, 1e-6, None)
    sel = (wv > 0) & (pmu_f > 250) & (cmu > -0.6) & (ppm > 450) & (ppm < 1000) & (cp > 0.4)
    dpt = np.asarray(obs.delta_pT(ev))[sel]
    dat = np.asarray(obs.delta_alphaT(ev))[sel]
    return dict(dpt=dpt, dalphat=dat, w=wv[sel])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    for fsi in (False, True):
        d = generate(n, with_fsi=fsi)
        tag = "FSI " if fsi else "noFSI"
        print(f"{tag}: {len(d['w'])} CC0pi-Np  <dpT>={np.average(d['dpt'], weights=d['w']):.1f}  "
              f"<daT>={np.degrees(np.average(d['dalphat'], weights=d['w'])):.1f}deg")
        np.savez(ROOT / "data" / "oracle" / f"t2k_cc0pi_tki_adonis_{'fsi' if fsi else 'nofsi'}.npz", **d)
    print("wrote data/oracle/t2k_cc0pi_tki_adonis_{fsi,nofsi}.npz")
