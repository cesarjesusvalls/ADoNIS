"""ADoNIS T2K CC1pi+ STV prediction: flux-folded CC single-pion events with the spectral-
function nuclear model + REAL pion (Oset+DCC) and nucleon (GiBUU NN-elastic) FSI, reduced to
the T2K STV observables (delta_pTT, p_N, delta_alphaT).

Signal: the produced pi+ survives FSI (pid 211) AND the leading nucleon is a proton (2212).
Returns weighted observable arrays for the no-FSI and with-FSI samples.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from adonis.flux.spectrum import Spectrum
from adonis.primary.dcc.channel import sample_final_state, weight_from_sample, assemble_event
from adonis.primary.dcc.amplitudes import DCCKnobs
from adonis.fsi.cascade_real import RealCascadeFSI, RealCascadeConfig
from adonis.fsi.nucleon_cascade import NucleonFSI, NucleonCascadeConfig
from adonis import observables as obs

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "_paper_release" / "AchillesGen-arXiv-2508.19213-48a9148"
M_MU = 105.6583745


def generate(n=400_000, seed=0, with_fsi=True):
    flux = Spectrum(REL / "T2K" / "T2K_nu.dat")
    key = jax.random.PRNGKey(seed)
    ke, ks, kpi, knuc = jax.random.split(key, 4)
    enu = np.asarray(flux.sample_enu(ke, n))
    S = sample_final_state(ks, n, e_nu=enu, ep_lo=M_MU + 10.0, ep_hi=enu, theta_max_deg=180.0,
                           m_lep=M_MU, current="CC", weight_ep_volume=True)
    w, LWc = weight_from_sample(DCCKnobs(), S)
    ev = assemble_event(S, w, LWc)
    if with_fsi:
        ev = RealCascadeFSI(RealCascadeConfig(seed=1, step=0.08, max_steps=160)).apply(None, ev, key=kpi)
        ev = NucleonFSI(NucleonCascadeConfig(seed=1, step=0.08, max_steps=160)).apply(None, ev, key=knuc)
    pid_pi = np.asarray(ev.pid_pi); pid_N = np.asarray(ev.pid_N)
    wv = np.asarray(ev.w)
    sel = (pid_pi == 211) & (pid_N == 2212) & (wv > 0)
    dptt = np.asarray(obs.delta_pTT(ev))[sel]
    pn = np.asarray(obs.p_N_tki(ev))[sel]
    dat = np.asarray(obs.delta_alphaT(ev))[sel]
    return dict(dptt=dptt, pn=pn, dalphat=dat, w=wv[sel])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
    for fsi in (False, True):
        d = generate(n, with_fsi=fsi)
        tag = "FSI " if fsi else "noFSI"
        wsum = d["w"].sum()
        print(f"{tag}: {len(d['w'])} CC1pi+  rms(dpTT)={np.sqrt(np.average(d['dptt']**2, weights=d['w'])):.1f}  "
              f"<p_N>={np.average(d['pn'], weights=d['w']):.1f}  <daT>={np.degrees(np.average(d['dalphat'], weights=d['w'])):.1f}deg")
        np.savez(ROOT / "data" / "oracle" / f"t2k_cc1pi_tki_adonis_{'fsi' if fsi else 'nofsi'}.npz", **d)
    print("wrote data/oracle/t2k_cc1pi_tki_adonis_{fsi,nofsi}.npz")
