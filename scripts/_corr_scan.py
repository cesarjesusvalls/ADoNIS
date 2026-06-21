"""Study the residual CORRELATION: extend the 'N_eff if perfectly importance-sampled' metric from
single invariants to PAIRS (2-D maps).  A pair whose joint gain >> its two singles carries exploitable
cross-variable structure (which separable Vegas / 1-D maps can't reach).  Uses a TRAIN/TEST split so
the conditional-mean estimate isn't overfit (the honest achievable N_eff, not a binning artifact).
Progress printed per stage."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from adonis.xsec import res_xsec as R
from adonis.xsec.flux import T2KFlux

t0 = time.time()
CACHE = "/tmp/corr_C.npz"
if os.path.exists(CACHE) and "regen" not in sys.argv:
    print(f"[1-2/4] loading cached events {CACHE}", flush=True)
    C = dict(np.load(CACHE))
else:
    R.SAMPLER_3BODY = "resonance"
    print(f"[1/4] warming up grid...", flush=True)
    grid = R.warmup_vegas(n=100000, iters=6, nbins=50, alpha=1.5, progress=False)
    print(f"      grid built ({time.time()-t0:.0f}s)", flush=True)
    rng = np.random.default_rng(0); flux = T2KFlux(); minE = flux.seed_min_GeV()
    cols = {k: [] for k in ("w","Q2","W","s_muN","s_mupi","t_N","t_pi","cmu","ppi","pN")}
    for ci, (ipid, itiz, mNf, ppid, mpi, mstr) in enumerate(R.CHANNELS):
        print(f"[2/4] gen+amps2 channel {ci+1}/3 (300k)...", flush=True)
        s = R._sample_channel(300000, rng, flux, minE, flux.max_energy, R._pi_kin_mass(mpi), mNf, grid=grid)
        w = R._channel_weight(s, ipid, itiz, ppid, mstr, R._SF_N, R._SF_P, 6, 6); k = w > 0
        knu,kmu,pst,pN,ppi = (s[x][k] for x in ("k_nu","k_mu","p_struck","p_N","p_pi"))
        def m2(p): return p[:,0]**2 - np.sum(p[:,1:]**2,1)
        q = knu-kmu
        cols["w"].append(w[k]); cols["Q2"].append(-m2(q)/1e6)
        cols["W"].append(np.sqrt(np.clip(m2(pN+ppi),0,None)))
        cols["s_muN"].append(np.sqrt(np.clip(m2(kmu+pN),0,None)))
        cols["s_mupi"].append(np.sqrt(np.clip(m2(kmu+ppi),0,None)))
        cols["t_N"].append(-m2(pst-pN)/1e6); cols["t_pi"].append(-m2(q-ppi)/1e6)
        cols["cmu"].append(kmu[:,3]/np.clip(np.linalg.norm(kmu[:,1:],axis=1),1e-9,None))
        cols["ppi"].append(np.linalg.norm(ppi[:,1:],axis=1)); cols["pN"].append(np.linalg.norm(pN[:,1:],axis=1))
    C = {k: np.concatenate(v) for k,v in cols.items()}
    np.savez(CACHE, **C); print(f"      cached events -> {CACHE}", flush=True)
w = C["w"]; n = len(w); rs = np.random.default_rng(1); perm = rs.permutation(n)
tr, te = perm[:n//2], perm[n//2:]
def neffN(x): x=x[x>0]; return x.sum()**2/np.sum(x**2)/len(x)
cur = neffN(w[te])
def binidx(arrs, nb, ref):
    idx = np.zeros(len(arrs[0]), int)
    for a, full in zip(arrs, ref):
        e = np.quantile(full, np.linspace(0,1,nb+1)); e[0]-=1e-9; e[-1]+=1e-9
        idx = idx*nb + np.clip(np.digitize(a, e)-1, 0, nb-1);
    return idx
def gain(names, nb):
    ref_tr = [C[x][tr] for x in names]; ref_te = [C[x][te] for x in names]
    nbin = nb**len(names)
    itr = binidx(ref_tr, nb, ref_tr); ite = binidx(ref_te, nb, ref_tr)
    sw = np.zeros(nbin); cnt = np.zeros(nbin); np.add.at(sw, itr, w[tr]); np.add.at(cnt, itr, 1.0)
    Ew = sw/np.clip(cnt,1,None)
    ew_te = np.where(cnt[ite] > 0, Ew[ite], np.median(Ew[Ew>0]))
    return w[te].mean()/np.mean(w[te]**2/np.clip(ew_te,1e-300,None))
invs = ["Q2","W","s_muN","s_mupi","t_N","t_pi","cmu","ppi","pN"]
print(f"[3/4] singles (train/test, nb=40)...", flush=True)
sg = {x: gain([x],40) for x in invs}
print(f"      current N_eff/N(test)={cur:.3f}   singles: " + ", ".join(f"{x} {sg[x]:.3f}" for x in invs), flush=True)
print(f"[4/4] pairs (train/test, nb=14)...", flush=True)
res = []
for i in range(len(invs)):
    for j in range(i+1,len(invs)):
        g = gain([invs[i],invs[j]], 14); res.append((g, invs[i], invs[j]))
res.sort(reverse=True)
print(f"\ncurrent N_eff/N = {cur:.3f}   (global upper bound w-self = {gain(['w'],40):.3f})")
print("top pairs (joint N_eff/N if perfectly 2-D mapped, vs the best of its two singles):")
for g, a, b in res[:10]:
    print(f"  {a:7s} x {b:7s}: {g:.3f}   (singles {sg[a]:.3f},{sg[b]:.3f}; pair gain x{g/max(sg[a],sg[b]):.2f})", flush=True)
