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
import jax.numpy as jnp

from adonis.flux.spectrum import Spectrum
from adonis.primary.dcc.channel import sample_final_state, weight_from_sample, assemble_event
from adonis.primary.dcc.amplitudes import DCCKnobs
from adonis.nuclear.spectral import SpectralFunction
from adonis.fsi.cascade_discrete import DiscreteCascadeFSI, DiscreteCascadeConfig
from adonis.fsi.nucleon_cascade import NucleonFSI, NucleonCascadeConfig
from adonis import observables as obs

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "_paper_release" / "AchillesGen-arXiv-2508.19213-48a9148"
M_MU = 105.6583745
# T2K target is CH: ACHILLES generates on 12C + 1H. Free-hydrogen CC1pi+ (nu p -> mu Delta++ ->
# mu p pi+) has NO Fermi motion -> a sharp p_N~0 / delta_pTT~0 peak. The measured H fraction in
# the ACHILLES CC1pi+ sample is ~0.25 (struck |p|<10 MeV); we reproduce that target composition.
H_FRACTION = 0.25


class _FreeNucleon:
    """Nuclear model for free hydrogen: a proton at rest (no Fermi momentum, no removal energy)."""
    def sample_nucleon(self, key, n):
        return jnp.zeros((n, 3)), jnp.zeros((n,))


def _gen_one(key, n, e_nu, nuclear, with_fsi, do_pion_fsi, do_nucleon_fsi):
    ks, kpi, knuc = jax.random.split(key, 3)
    S = sample_final_state(ks, n, e_nu=e_nu, ep_lo=M_MU + 10.0, ep_hi=e_nu, theta_max_deg=180.0,
                           m_lep=M_MU, current="CC", weight_ep_volume=True, nuclear=nuclear)
    w, LWc = weight_from_sample(DCCKnobs(), S)
    ev = assemble_event(S, w, LWc)
    if with_fsi and do_pion_fsi:
        ev = DiscreteCascadeFSI(DiscreteCascadeConfig(seed=1, step=0.05, max_steps=260)).apply(None, ev, key=kpi)
    if with_fsi and do_nucleon_fsi:
        ev = NucleonFSI(NucleonCascadeConfig(seed=1, step=0.08, max_steps=160)).apply(None, ev, key=knuc)
    pid_pi = np.asarray(ev.pid_pi); pid_N = np.asarray(ev.pid_N); wv = np.asarray(ev.w)
    sel = (pid_pi == 211) & (pid_N == 2212) & (wv > 0)
    return (np.asarray(obs.delta_pTT(ev))[sel], np.asarray(obs.p_N_tki(ev))[sel],
            np.asarray(obs.delta_alphaT(ev))[sel], wv[sel])


def generate(n=400_000, seed=0, with_fsi=True):
    """CH target: a carbon sub-sample (spectral fn + pion & nucleon FSI) and a free-hydrogen
    sub-sample (proton at rest, no nuclear FSI -- only the pion can rescatter on... nothing, so
    no FSI), combined at the ACHILLES H fraction."""
    flux = Spectrum(REL / "T2K" / "T2K_nu.dat")
    kc, kh = jax.random.split(jax.random.PRNGKey(seed))
    nuc = SpectralFunction("pke12p_tot.data")
    nC = int(round(n * (1 - H_FRACTION))); nH = n - nC
    eC = np.asarray(flux.sample_enu(jax.random.fold_in(kc, 0), nC))
    eH = np.asarray(flux.sample_enu(jax.random.fold_in(kh, 0), nH))
    dC = _gen_one(kc, nC, eC, nuc, with_fsi, True, True)
    dH = _gen_one(kh, nH, eH, _FreeNucleon(), with_fsi, False, False)   # free p: no nuclear FSI
    # scale the two sub-samples so the H weight fraction = H_FRACTION
    wC, wH = dC[3], dH[3]
    sC, sH = wC.sum(), wH.sum()
    if sC > 0 and sH > 0:
        wH = wH * (H_FRACTION / (1 - H_FRACTION)) * (sC / sH)
    cat = lambda a, b: np.concatenate([a, b])
    return dict(dptt=cat(dC[0], dH[0]), pn=cat(dC[1], dH[1]),
                dalphat=cat(dC[2], dH[2]), w=cat(wC, wH))


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
