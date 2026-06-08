"""sigma_RES the EXACT-ACHILLES way: flat QESpectralMapper struck nucleon (explicit N*S + J_had,
ACHILLES bounds) + exact ThreeBodyMapper forward sampler + the bit-exact ThreeBodyGenerateWeight.
NO importance sampling, NO isotropic shortcut.  This is ACHILLES's calculation minus Vegas.
If sigma ~ ACHILLES, the importance/isotropic res_xsec has a hidden bug; if ~0.74, deeper.
Neutron channel n->n pi+ (uses pke12n -> no proton-SF ambiguity)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
from adonis.xsec import constants as C
from adonis.xsec.flux import T2KFlux
from adonis.xsec.spectral import SpectralFunction
from adonis.xsec.backend import flux_factor, MASS_PDG_NEUTRON
from adonis.xsec.dcc_current import exclusive_amps2_batch
from scripts.res_tchannel_sigma import tchannel_momenta, iso2_momenta, sqlam
from scripts.validate_res_psw import three_body_genweight

MN=C.mN; M_MU=105.7; M_N=939.57; M_PI=139.57018; N_NUC=6; SPIN=0.5; TWO_PI=2*np.pi
_SFn=SpectralFunction("data/Spectral_Functions/pke12n_tot.data")

def sigma_exact(n, seed):
    m_Nf, mpi, itiz, ppid = M_N, M_PI, -1, 211     # n -> n pi+
    rng=np.random.default_rng(seed); u=rng.random((n,7))
    flux=T2KFlux(); maxE=flux.max_energy
    Smin=(M_MU+m_Nf+mpi)**2
    minE=max((Smin-m_Nf**2)/(2*m_Nf)/1000.0, flux.min_energy)
    Enu=(u[:,0]*(maxE-minE)+minE)*1000.0
    k_nu=np.stack([Enu,np.zeros(n),np.zeros(n),Enu],axis=1)
    J_beam=((maxE-minE)*flux.f(Enu/1000.0))/flux.flux_integral
    # FLAT struck nucleon, ACHILLES QESpectralMapper bounds (Smin=3-body), explicit N*S + J_had
    radical=np.clip(Enu**2+2*Enu*MN+MN**2-Smin,0,None)
    pmin=np.clip(Enu-np.sqrt(radical),0,None); pmax=np.clip(Enu+np.sqrt(radical),None,800.0)
    dp=pmax-pmin; mom=dp*u[:,1]+pmin
    cosTm=np.clip((2*Enu*MN+MN**2-mom**2-Smin)/(2*Enu*np.clip(mom,1e-9,None)),-1,1)
    cosT=(cosTm+1)*u[:,2]-1; sinT=np.sqrt(np.clip(1-cosT**2,0,None)); phi=TWO_PI*u[:,3]
    pvec=np.stack([mom*sinT*np.cos(phi),mom*sinT*np.sin(phi),mom*cosT],axis=1)
    det=Enu**2+mom**2+2*pvec[:,2]*Enu+Smin
    emax=MN+Enu-np.sqrt(np.clip(det,0,None)); emax=np.minimum(np.minimum(emax,MN-mom),400.0)
    energy=emax*u[:,4]-1e-8
    p_struck=np.concatenate([(MN-energy)[:,None],pvec],axis=1)
    J_had=mom**2*dp*(cosTm+1)*TWO_PI*emax
    initwgt=N_NUC*_SFn.batch(mom,energy)
    P=k_nu+p_struck; s=P[:,0]**2-np.sum(P[:,1:]**2,axis=1); sqrts=np.sqrt(np.clip(s,1e-9,None))
    s23min=(M_MU+m_Nf)**2; s23max=(sqrts-mpi)**2; s23=s23min+(s23max-s23min)*u[:,5]
    # EXACT ThreeBodyMapper forward: total->(muN,s23)+pi via tchannel, (muN)->mu+N iso
    p23,p_pi=tchannel_momenta(k_nu,p_struck,s23,mpi**2,u[:,6],rng.random(n))
    k_mu,p_N=iso2_momenta(p23,M_MU**2,m_Nf**2,rng.random(n),rng.random(n))
    gw=np.array([three_body_genweight(p_struck[i],k_nu[i],p_N[i],p_pi[i],k_mu[i]) for i in range(n)])
    valid=(dp>0)&(emax>0)&(s>Smin)&(s23max>s23min)&(gw>0)&(energy<emax)
    a2=np.zeros(n); idx=np.where(valid)[0]
    if len(idx):
        a2[idx]=exclusive_amps2_batch(k_nu[idx],k_mu[idx],p_struck[idx],p_N[idx],p_pi[idx],itiz,ppid)
    fl=np.asarray(flux_factor(k_nu,p_struck,had_mass=MASS_PDG_NEUTRON))
    w=np.where(valid&(a2>0),a2*fl*initwgt*SPIN*J_beam*J_had/np.clip(gw,1e-300,None),0.0)
    w=np.where(np.isfinite(w),w,0.0)
    return w.mean()

if __name__=="__main__":
    n=int(sys.argv[1]) if len(sys.argv)>1 else 60000
    sigs=np.array([sigma_exact(n,seed=s) for s in range(6)])
    ACH_NPIP=2.178e-6  # ACHILLES n->n pi+ channel
    print("n->n pi+ EXACT-ACHILLES (flat struck + exact 3body + bit-exact weight):")
    print("  sigma = %.4e  sem %.2e  vs ACHILLES %.3e  ratio %.3f"%(sigs.mean(),sigs.std()/np.sqrt(6),ACH_NPIP,sigs.mean()/ACH_NPIP))
    print("  (res_xsec importance n->npi+ = 1.585e-6 = 0.728)")
