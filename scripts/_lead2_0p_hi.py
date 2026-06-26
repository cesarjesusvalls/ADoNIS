"""0p residual verdict at 4x stats (lead2hi). Among absorbed-RES (CC0pi meson-veto, muon window):
  (1) proton multiplicity (>250) 0..4
  (2) FULL leading-proton |p| spectrum (bulk + tail) -> global softness vs tail-only?
  (3) mean leading |p| of the 0p(>250) events."""
import sys, numpy as np
RB = sys.argv[1] if len(sys.argv) > 1 else "data/oracle/t2k_cc1pi_lead2hi.npz"
REF = "data/oracle/t2k_cc1pi_rich_ach_FSI_proc.npz"

def absorbed(bank, ref=False):
    d = np.load(bank, allow_pickle=True)
    mu = d["mu"]; w = d["w"].astype(float); struck = d["struck"]
    pm = np.linalg.norm(mu[:, 1:], axis=1); cth = mu[:, 3] / np.clip(pm, 1e-9, None)
    sel = (pm > 250.0) & (cth > -0.6) & (w > 0) & (np.linalg.norm(struck[:, 1:], axis=1) > 1.0)
    if ref:
        proc = d["proc"]; sel = sel & ((proc == 401) | (proc == 402)); w = w * float(d["weight_to_nb"])
        npi = (np.linalg.norm(d["pi_p4"][:, :, 1:], axis=2) > 1.0).sum(1); prot = d["prot_p4"]
    else:
        npi = (np.linalg.norm(d["pi_post"][:, 1:], axis=1) > 1.0).astype(int); prot = d["prot"]
    absb = sel & (npi == 0)
    return w[absb], prot[absb]

wa, pa = absorbed(RB); wr, pr = absorbed(REF, ref=True)
Wa, Wr = wa.sum(), wr.sum()
print(f"absorbed-RES: ADO sigma={Wa:.4e} N={len(wa)}   ACH sigma={Wr:.4e} N={len(wr)}  ACH/ADO={Wr/Wa:.4f}")

def mult(p, thr): return (np.linalg.norm(p[:, :, 1:], axis=2) > thr).sum(1)
na, nr = mult(pa, 250.0), mult(pr, 250.0)
print("\n(1) n_p(>250) multiplicity:")
print(f"{'k':>2s} {'ADO':>8s} {'ACH':>8s} {'ACH/ADO':>8s} {'Nado':>6s}")
for k in range(5):
    fa = wa[na == k].sum()/Wa; fr = wr[nr == k].sum()/Wr
    print(f"{k:2d} {fa:8.4f} {fr:8.4f} {fr/fa if fa else float('nan'):8.3f} {int((na==k).sum()):6d}")

la = np.linalg.norm(pa[:, :, 1:], axis=2).max(1); lr = np.linalg.norm(pr[:, :, 1:], axis=2).max(1)
print("\n(2) FULL leading-proton |p| spectrum (frac of absorbed):")
edges = [0,175,225,250,275,325,400,500,650,850,1100,1500]
print(f"{'bracket':12s} {'ADO':>8s} {'ACH':>8s} {'ACH/ADO':>8s} {'Nado':>6s}")
for lo, hi in zip(edges[:-1], edges[1:]):
    ma = (la >= lo) & (la < hi); mr = (lr >= lo) & (lr < hi)
    fa = wa[ma].sum()/Wa; fr = wr[mr].sum()/Wr
    print(f"[{lo:4d},{hi:4d}) {fa:8.4f} {fr:8.4f} {fr/fa if fa else float('nan'):8.3f} {int(ma.sum()):6d}")
# weighted mean leading |p| (global) and of the 0p(>250) subset
print(f"\n(3) mean leading |p|  ADO={ (wa*la).sum()/Wa:7.1f}  ACH={ (wr*lr).sum()/Wr:7.1f}  (global)")
za = na == 0; zr = nr == 0
print(f"    0p(>250) leading|p| mean  ADO={(wa[za]*la[za]).sum()/wa[za].sum() if za.any() else 0:7.1f} (N={int(za.sum())})"
      f"  ACH={(wr[zr]*lr[zr]).sum()/wr[zr].sum() if zr.any() else 0:7.1f} (N={int(zr.sum())})")
