"""Screen for any other 'BW-analogous' trick: a sharp structure in a physical invariant that isn't a
sampling axis (so Vegas misses it).  For each candidate invariant X, estimate N_eff/N achievable if X
were PERFECTLY importance-sampled (a BW-style 1-D map on X):
    N_eff_X/N = mean(w) / mean(w^2 / E[w|X])     (E[w|X] = binned conditional mean)
Current N_eff/N = mean(w)/ (mean(w^2)/mean(w)).  A big jump for some X => a missed mappable structure.
Controls: W(Npi) should show ~no gain (already BW-mapped); Enu/struck|p| ~none (already importance-
sampled); X=w is the trivial upper bound (->1)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from adonis.xsec import res_xsec as R

import time as _t
R.SAMPLER_3BODY = "resonance"
print("[1/5] warming up Vegas grid (100k x 6)...", flush=True); _t0 = _t.time()
grid = R.warmup_vegas(n=100000, iters=6, nbins=50, alpha=1.5, progress=False)
print(f"      grid built ({_t.time()-_t0:.0f}s)", flush=True)
rng = np.random.default_rng(0)
from adonis.xsec.flux import T2KFlux
from adonis.xsec.backend import flux_factor
from adonis.xsec.dcc_current import exclusive_amps2_batch
flux = T2KFlux(); minE = flux.seed_min_GeV()
W = {}; cols = {k: [] for k in ("w","Q2","Wnpi","s_muN","s_mupi","t_N","t_pi","cmu","ppi","pN","Enu","pstr")}
for _ci, (ipid, itiz, mNf, ppid, mpi, mstr) in enumerate(R.CHANNELS):
    print(f"[{2+_ci}/5] generating + amps2 channel {_ci+1}/3 (150k)...", flush=True); _t1 = _t.time()
    s = R._sample_channel(150000, rng, flux, minE, flux.max_energy, R._pi_kin_mass(mpi), mNf, grid=grid)
    w = R._channel_weight(s, ipid, itiz, ppid, mstr, R._SF_N, R._SF_P, 6, 6)
    print(f"      channel {_ci+1} done ({_t.time()-_t1:.0f}s)", flush=True)
    k = w > 0
    knu,kmu,pst,pN,ppi = (s[x][k] for x in ("k_nu","k_mu","p_struck","p_N","p_pi"))
    def m2(p): return p[:,0]**2 - np.sum(p[:,1:]**2,1)
    q = knu-kmu
    cols["w"].append(w[k])
    cols["Q2"].append(-m2(q)/1e6)
    cols["Wnpi"].append(np.sqrt(np.clip(m2(pN+ppi),0,None)))
    cols["s_muN"].append(np.sqrt(np.clip(m2(kmu+pN),0,None)))
    cols["s_mupi"].append(np.sqrt(np.clip(m2(kmu+ppi),0,None)))
    cols["t_N"].append(-m2(pst-pN)/1e6)
    cols["t_pi"].append(-m2(q-ppi)/1e6)
    cols["cmu"].append(kmu[:,3]/np.clip(np.linalg.norm(kmu[:,1:],axis=1),1e-9,None))
    cols["ppi"].append(np.linalg.norm(ppi[:,1:],axis=1))
    cols["pN"].append(np.linalg.norm(pN[:,1:],axis=1))
    cols["Enu"].append(knu[:,0])
    cols["pstr"].append(np.linalg.norm(pst[:,1:],axis=1))
C = {k: np.concatenate(v) for k,v in cols.items()}
w = C["w"]
def neffN(x): x=x[x>0]; return x.sum()**2/np.sum(x**2)/len(x)
def gain_if_mapped(X, nbins=60):
    e = np.quantile(X, np.linspace(0,1,nbins+1)); e[0]-=1e-9; e[-1]+=1e-9
    idx = np.clip(np.digitize(X,e)-1,0,nbins-1)
    sw = np.zeros(nbins); cnt = np.zeros(nbins)
    np.add.at(sw,idx,w); np.add.at(cnt,idx,1.0)
    Ewx = (sw/np.clip(cnt,1,None))[idx]                 # E[w|X] per event
    return w.mean()/np.mean(w**2/np.clip(Ewx,1e-300,None))
cur = neffN(w)
print("[5/5] screening invariants...", flush=True)
print(f"current resonance+Vegas N_eff/N = {cur:.3f}\n")
print(f"{'invariant':10s} {'N_eff/N if perfectly mapped':>26s}   gain")
for nm in ("w","Wnpi","Enu","pstr","Q2","s_muN","s_mupi","t_N","t_pi","cmu","ppi","pN"):
    import sys; g = gain_if_mapped(C[nm])
    tag = "  <-- TRIVIAL upper bound" if nm=="w" else ("  (control: already handled)" if nm in ("Wnpi","Enu","pstr") else "")
    print(f"{nm:10s} {g:26.3f}   x{g/cur:.2f}{tag}", flush=True)
