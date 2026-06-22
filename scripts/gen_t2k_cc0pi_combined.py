"""ADoNIS T2K CC0pi-Np STV prediction, FAITHFUL to how ACHILLES builds the sample:
the CC0pi-Np signal is NOT pure CCQE.  ACHILLES generates QE (process 200) AND RES (401/402)
at their own absolute cross sections, runs the cascade, then classifies the FINAL STATE -- any
event with no surviving meson is CC0pi.  Resonance events whose pion is ABSORBED in the nucleus
(piNN->NN, PionAbsorption.cc) therefore enter CC0pi, producing two energetic nucleons that
populate the high-delta_pT tail (paper line 590: "pion absorption ... leads to an increase of
events in the 0pi sample").

Composition is NOT a fitted fraction.  Per XSecBackend.cc both processes share the identical
FluxFactor x InitialStateWeight x SpinAvg on a common nb scale; the QE:RES mix is fixed by the
two absolute cross sections, which ADoNIS computes from its OWN validated models:
  - QE  : Llewellyn-Smith dsigma/dQ^2 [nb]  (Fig 1)
  - RES : DCC weight x SIGMA_UNIT_NB [nb]   (Fig 2, one universal bridge constant)
Both are per single target nucleon; the 12C nucleon-count factor (6) is common to QE (neutrons)
and RES (p+n, symmetric C) and cancels in the area-normalised shape, so concatenating each
sample at its own absolute weight reproduces the ACHILLES mix with NO tuning to the Fig-7 shape.

CC0pi-Np cuts (NUISANCE T2K_CC0pi_STV): p_mu>250, cos_mu>-0.6, leading proton 450-1000 MeV/c
with cos_p>0.4, no mesons.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.flux.spectrum import Spectrum
from adonis.primary.dcc.channel import sample_final_state, weight_from_sample, assemble_event
from adonis.primary.dcc.amplitudes import DCCKnobs
from adonis.primary.dcc.sigma_enu import SIGMA_UNIT_NB
from adonis.primary.qe.llewellyn_smith import ls_dsigma_dQ2
from adonis.nuclear.spectral import SpectralFunction
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig
from adonis.fsi.pool_fsi import run_fsi
from adonis import observables as obs
from adonis.core.event import EventRecord

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "_paper_release" / "AchillesGen-arXiv-2508.19213-48a9148"
M_MU = 105.6583745
M_N = 939.0          # MeV (mean nucleon)
M_P = 938.272

# CC0pi-Np tight phase space
MU_LO = 250.0; COSMU = -0.6; P_LO, P_HI = 450.0, 1000.0; COSP = 0.4
# the joint Gaussian pool is the single differentiable cascade engine (cylinder honored until removed)
_CFG = lambda **k: DiscreteCascadeConfig(step=0.04, max_steps=600, engine="pool", **k)


def _qe_cc0pi(key, n, flux, nuc, MA_GeV=1.03):
    """CCQE (Llewellyn-Smith off S(p,E)) CC0pi-Np sub-sample with proton FSI through the pool.
    Inlined from the (retired) gen_t2k_cc0pi_adonis QE generator; FSI now via the pool engine."""
    ke, kq, ksf, kph, knuc = jax.random.split(key, 5)
    enu = np.asarray(flux.sample_enu(ke, n))                      # MeV
    Q2hi = np.minimum(2.0 * M_N * enu, 3.0e6)                     # MeV^2 cap
    Q2 = np.asarray(jax.random.uniform(kq, (n,))) * Q2hi
    dsig = np.asarray(ls_dsigma_dQ2(Q2 / 1e6, enu / 1000.0, MA=MA_GeV, m_l=M_MU / 1000.0))
    wq = np.clip(dsig, 0.0, None) * (Q2hi / 1e6)                  # x proposal volume [GeV^2]
    omega = Q2 / (2.0 * M_N); Emu = enu - omega
    pmu = np.sqrt(np.clip(Emu ** 2 - M_MU ** 2, 0.0, None))
    cth = (2.0 * enu * Emu - Q2 - M_MU ** 2) / (2.0 * enu * np.clip(pmu, 1e-6, None))
    valid = (Emu > M_MU) & (np.abs(cth) <= 1.0) & (wq > 0)
    cth = np.clip(cth, -1.0, 1.0); sth = np.sqrt(np.clip(1 - cth ** 2, 0.0, None))
    phi = np.asarray(jax.random.uniform(kph, (n,))) * 2 * np.pi
    kmu = np.stack([Emu, pmu * sth * np.cos(phi), pmu * sth * np.sin(phi), pmu * cth], axis=1)
    knu = np.stack([enu, np.zeros(n), np.zeros(n), enu], axis=1)
    q = knu - kmu
    p_vec, E_rm = nuc.sample_nucleon(ksf, n); p_vec = np.asarray(p_vec)
    p_p3 = p_vec + q[:, 1:]; Ep = np.sqrt(M_P ** 2 + np.sum(p_p3 ** 2, axis=1))
    p_p = np.concatenate([Ep[:, None], p_p3], axis=1)
    w = np.where(valid, wq, 0.0)
    ev = EventRecord(k=jnp.asarray(knu), kp=jnp.asarray(kmu), p_struck=jnp.asarray(
        np.concatenate([(M_N - np.asarray(E_rm))[:, None], p_vec], axis=1)),
        p_pi=jnp.zeros((n, 4)), p_N=jnp.asarray(p_p), w=jnp.asarray(w),
        channel=jnp.zeros(n, jnp.int32), pid_pi=jnp.zeros(n, jnp.int32),
        pid_N=jnp.full((n,), 2212), pid_Ni=jnp.full((n,), 2112),
        W=jnp.zeros(n), Q2_adj=jnp.asarray(Q2))
    out = run_fsi(jnp.zeros((n, 4)), jnp.asarray(p_p), jnp.zeros(n, jnp.int32),
                  jnp.full(n, 2112, jnp.int32), jnp.full(n, 2212, jnp.int32),
                  _CFG(seed=1), knuc, channel="qe")
    lead = np.asarray(out["lead_prot"])
    ev = ev._replace(p_N=jnp.asarray(lead))
    mu = np.asarray(kmu); wv = w
    pmu_f = np.linalg.norm(mu[:, 1:], axis=1); cmu = mu[:, 3] / np.clip(pmu_f, 1e-6, None)
    ppm = np.linalg.norm(lead[:, 1:], axis=1); cp = lead[:, 3] / np.clip(ppm, 1e-6, None)
    sel = (wv > 0) & (pmu_f > MU_LO) & (cmu > COSMU) & (ppm > P_LO) & (ppm < P_HI) & (cp > COSP)
    dpt = np.asarray(obs.delta_pT(ev))[sel]; dat = np.asarray(obs.delta_alphaT(ev))[sel]
    return dict(dpt=dpt, dalphat=dat, w=wv[sel])


def _res_absorbed(key, n, e_nu, nuclear):
    """Generate RES (CC1pi) events, propagate the pion (discrete-Glauber); KEEP the events whose
    pion is ABSORBED (piNN->NN) -> they are CC0pi.  The leading proton is the highest-momentum
    proton among {primary recoil nucleon after NN-elastic FSI, the absorption proton}.  Returns
    (dpt, dat, w_nb) after the CC0pi-Np cuts."""
    ks, kpi = jax.random.split(key, 2)
    S = sample_final_state(ks, n, e_nu=e_nu, ep_lo=M_MU + 10.0, ep_hi=e_nu, theta_max_deg=180.0,
                           m_lep=M_MU, current="CC", weight_ep_volume=True, nuclear=nuclear)
    w, LWc = weight_from_sample(DCCKnobs(), S)
    ev = assemble_event(S, w, LWc)

    # joint pool cascade (pion + recoil nucleon): absorbed-pion events are CC0pi; the pool re-cascades
    # both absorption nucleons, so lead_prot is the leading escaped proton among all candidates.
    out = run_fsi(ev.p_pi, ev.p_N, ev.pid_pi, ev.pid_Ni, ev.pid_N, _CFG(seed=1), kpi, channel="res")
    absorbed = np.asarray(out["pterm"]["pid"] == 0)
    lead = np.asarray(out["lead_prot"])
    mu = np.asarray(ev.kp); wv = np.asarray(ev.w)
    has_proton = np.linalg.norm(lead[:, 1:], axis=1) > 1

    # build CC0pi event: pion gone, p_N = leading proton
    n_ = len(wv)
    ev0 = ev._replace(p_pi=jnp.zeros((n_, 4)), p_N=jnp.asarray(lead))
    dpt = np.asarray(obs.delta_pT(ev0)); dat = np.asarray(obs.delta_alphaT(ev0))

    pmu = np.linalg.norm(mu[:, 1:], axis=1); cmu = mu[:, 3] / np.clip(pmu, 1e-9, None)
    plead = np.linalg.norm(lead[:, 1:], axis=1); clead = lead[:, 3] / np.clip(plead, 1e-9, None)
    sel = (absorbed & has_proton & (wv > 0) & (pmu > MU_LO) & (cmu > COSMU)
           & (plead > P_LO) & (plead < P_HI) & (clead > COSP))
    return dpt[sel], dat[sel], wv[sel] * SIGMA_UNIT_NB        # nb


def generate(n_qe=1_000_000, n_res=1_500_000, seed=0, with_fsi=True):
    flux = Spectrum(REL / "T2K" / "T2K_nu.dat")
    nuc = SpectralFunction("pke12p_tot.data")
    # QE sub-sample (Llewellyn-Smith, nb weights), proton FSI through the pool
    q = _qe_cc0pi(jax.random.PRNGKey(seed), n_qe, flux, nuc)
    # RES-absorbed sub-sample (DCC x SIGMA_UNIT_NB, nb)
    kr = jax.random.PRNGKey(seed + 100)
    eR = np.asarray(flux.sample_enu(jax.random.fold_in(kr, 1), n_res))
    nuc_res = SpectralFunction("pke12p_tot.data")
    rdpt, rdat, rw = _res_absorbed(kr, n_res, eR, nuc_res)
    cat = lambda a, b: np.concatenate([a, b])
    out = dict(dpt=cat(q["dpt"], rdpt), dalphat=cat(q["dalphat"], rdat), w=cat(q["w"], rw))
    sQE, sRES = q["w"].sum(), rw.sum()
    out["_sig_qe"] = sQE; out["_sig_res"] = sRES
    return out


if __name__ == "__main__":
    n_qe = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    n_res = int(sys.argv[2]) if len(sys.argv) > 2 else 1_500_000
    d = generate(n_qe, n_res, with_fsi=True)
    f_res = d["_sig_res"] / (d["_sig_qe"] + d["_sig_res"])
    print(f"combined CC0pi-Np: {len(d['w'])} events  "
          f"sigma_QE={d['_sig_qe']:.4e}  sigma_RES_abs={d['_sig_res']:.4e} nb  "
          f"RES-absorbed fraction of CC0pi = {100*f_res:.1f}%")
    print(f"  <dpT>={np.average(d['dpt'], weights=d['w']):.1f} MeV  "
          f"<daT>={np.degrees(np.average(d['dalphat'], weights=d['w'])):.1f} deg")
    np.savez(ROOT / "data" / "oracle" / "t2k_cc0pi_tki_adonis_combined.npz",
             dpt=d["dpt"], dalphat=d["dalphat"], w=d["w"])
    print("wrote data/oracle/t2k_cc0pi_tki_adonis_combined.npz")
