"""Understand the absorbed-RES 0p residual (ADoNIS over-produces zero-proton absorbed-RES events).
Among CC0pi.RES events (meson veto + muon window, NO proton cut), on both sides:
  - full proton-multiplicity distribution (n protons >250 MeV): 0,1,2,3,4
  - composition of the 0p events: how many protons of ANY momentum, leading-proton |p|
ADoNIS lead2 (neutrons re-cascaded) vs ACHILLES-C FSI proc 401/402."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

RB = "data/oracle/t2k_cc1pi_engine_rich_lead2.npz"
REF = "data/oracle/t2k_cc1pi_rich_ach_FSI_proc.npz"

def absorbed_sample(bank, ref=False):
    d = np.load(bank, allow_pickle=True)
    mu = d["mu"]; w = d["w"].astype(float); struck = d["struck"]
    pm = np.linalg.norm(mu[:, 1:], axis=1); cth = mu[:, 3] / np.clip(pm, 1e-9, None)
    sel = (pm > 250.0) & (cth > -0.6) & (w > 0) & (np.linalg.norm(struck[:, 1:], axis=1) > 1.0)
    if ref:
        proc = d["proc"]; sel = sel & ((proc == 401) | (proc == 402))
        w = w * float(d["weight_to_nb"])
        # absorbed = no pion of any kind survives
        pim = np.linalg.norm(d["pi_p4"][:, :, 1:], axis=2)        # (N,4)
        npi = (pim > 1.0).sum(1)
        prot = d["prot_p4"]                                       # (N,?,4)
    else:
        pim = np.linalg.norm(d["pi_post"][:, 1:], axis=1)         # (N,)
        npi = (pim > 1.0).astype(int)
        prot = d["prot"]                                          # (N,4,4)
    absb = sel & (npi == 0)
    return w, absb, prot

def proton_mult(prot, thr):
    return (np.linalg.norm(prot[:, :, 1:], axis=2) > thr).sum(1)

for name, bank, ref in [("ADO", RB, False), ("ACH", REF, True)]:
    w, absb, prot = absorbed_sample(bank, ref)
    W = w[absb].sum(); N = int(absb.sum())
    print(f"\n=== {name}: absorbed-RES total sigma={W:.4e}  N={N} ===")
    nej = proton_mult(prot[absb], 250.0)
    nall = proton_mult(prot[absb], 1.0)
    ww = w[absb]
    print("  n_p(>250):  " + "  ".join(f"{k}p={ww[nej==k].sum()/W:.4f}" for k in range(5)))
    print("  n_p(>1):    " + "  ".join(f"{k}p={ww[nall==k].sum()/W:.4f}" for k in range(5)))
    # composition of the 0p(>250) events
    zp = absb.copy(); zp[absb] = (nej == 0)
    wz = w[zp]; Wz = wz.sum()
    if Wz > 0:
        nall_z = proton_mult(prot[zp], 1.0)            # protons of ANY momentum in 0p events
        lead = np.linalg.norm(prot[zp][:, :, 1:], axis=2).max(1)   # leading proton |p| (<=250)
        print(f"  0p(>250) sigma={Wz:.4e} ({Wz/W:.3f} of absorbed)  N={int(zp.sum())}")
        print("    of those, n_p(>1): " + "  ".join(f"{k}={wz[nall_z==k].sum()/Wz:.3f}" for k in range(4)))
        print(f"    leading-proton |p| (all <=250): mean={(wz*lead).sum()/Wz:6.1f}  frac|p|<1(zero prot)={wz[lead<1].sum()/Wz:.3f}")
