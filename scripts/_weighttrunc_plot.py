"""Usual CC1pi RES-only distributions (ADoNIS cv5 resonance+Vegas vs ACHILLES), but showing what the
top-weight events contribute: ADoNIS with (a) all weights, (b) top 0.1% weights dropped, (c) top 1%
dropped.  Dropping high weights BIASES the prediction low (removes cross section) -- the gap shows how
much each bin leans on its few biggest weights."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.workflow.config import load_analysis_config
import adonis.workflow.signal as SG

cfg = load_analysis_config("configs/ana_cc1pi_resonly_cv5.yaml")
ado = SG.select_signal(cfg.inputs["adonis_res"][0], cfg.signal, carbon_only=cfg.carbon_only)
ref = SG.select_reference(cfg.inputs["reference"][0], cfg.signal, carbon_only=cfg.carbon_only)
w = np.asarray(ado["w"]); rw = np.asarray(ref["w"])
pA, pB = np.percentile(w[w > 0], 99.95), np.percentile(w[w > 0], 99.99)
w_no1 = np.where(w > pA, 0.0, w); w_no01 = np.where(w > pB, 0.0, w)   # w_no1=drop>p99.95, w_no01=drop>p99.99
print(f"sel N={len(w)}  drop>p99.95: {(w>pA).sum()} ev ({(w[w>pA].sum()/w.sum()*100):.2f}% of sigma)  "
      f"drop>p99.99: {(w>pB).sum()} ev ({(w[w>pB].sum()/w.sum()*100):.2f}% of sigma)", flush=True)

specs = [(o.key, o.bin_edges(), o.label) for o in cfg.observables]
fig, axes = plt.subplots(2, len(specs), figsize=(3.0*len(specs), 6.0),
                         gridspec_kw={"height_ratios": [3, 1]})
for j, (key, ed, lab) in enumerate(specs):
    x = np.asarray(ado[key]); xr = np.asarray(ref[key]); c = 0.5*(ed[:-1]+ed[1:])
    hR = np.histogram(xr, ed, weights=rw)[0]
    hF = np.histogram(x, ed, weights=w)[0]
    h1 = np.histogram(x, ed, weights=w_no1)[0]
    h01 = np.histogram(x, ed, weights=w_no01)[0]
    a = axes[0, j]
    a.step(c, hR, where="mid", color="k", lw=1.6, label="ACHILLES")
    a.step(c, hF, where="mid", color="crimson", lw=1.8, label="ADoNIS all")
    a.step(c, h01, where="mid", color="darkorange", lw=1.4, ls="--", label="drop >p99.99")
    a.step(c, h1, where="mid", color="royalblue", lw=1.4, ls=":", label="drop >p99.95")
    a.set_title(lab, fontsize=9)
    if j == 0: a.legend(fontsize=7); a.set_ylabel("d$\\sigma$ [nb]")
    r = axes[1, j]
    r.step(c, hF/np.clip(hR,1e-30,None), where="mid", color="crimson", lw=1.5)
    r.step(c, h01/np.clip(hR,1e-30,None), where="mid", color="darkorange", lw=1.2, ls="--")
    r.step(c, h1/np.clip(hR,1e-30,None), where="mid", color="royalblue", lw=1.2, ls=":")
    r.axhline(1, ls=":", c="gray"); r.set_ylim(0.5, 1.3); r.set_xlabel(lab, fontsize=8)
    if j == 0: r.set_ylabel("ADO/ACH")
fig.suptitle("CC1$\\pi$ RES-only (C, resonance+Vegas): effect of dropping top-weight events", fontsize=11)
plt.tight_layout(); out = "paper_figures/cc1pi_weighttrunc_fine_C.png"; plt.savefig(out, dpi=120); print("wrote", out)
