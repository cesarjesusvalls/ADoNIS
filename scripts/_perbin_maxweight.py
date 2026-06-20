"""For each analysis config: after the ADoNIS signal selection, bin every observable and report the
single bin most dominated by ONE event -- max(w in bin)/sum(w in bin).  This is what makes a plotted
bin spike/noisy (a single high-weight event)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from adonis.workflow.config import load_analysis_config
from adonis.workflow.analyze import build_adonis

CONFIGS = [
    "ana_cc0pi_qeonly_c6h", "ana_cc0pi_resonly_c6h", "ana_cc0pi_c6h",
    "ana_cc1pi_qeonly_c6h", "ana_cc1pi_resonly_c6h", "ana_cc1pi_c6h",
    "ana_cc0pi_qeonly_ar6h", "ana_cc0pi_resonly_ar6h", "ana_cc0pi_ar6h",
    "ana_cc1pi_qeonly_ar6h", "ana_cc1pi_resonly_ar6h", "ana_cc1pi_ar6h",
]
print(f"{'config':28s} {'worst bin (obs)':14s} {'maxw/binsum':>11s} {'binN':>6s}")
for name in CONFIGS:
    cfg = load_analysis_config(f"configs/{name}.yaml")
    ado = build_adonis(cfg)
    w = np.asarray(ado["w"])
    worst = (0.0, "", 0)
    for o in cfg.observables:
        if o.key not in ado:
            continue
        x = np.asarray(ado[o.key]); ed = o.bin_edges()
        idx = np.digitize(x, ed) - 1
        for bi in range(len(ed) - 1):
            m = idx == bi
            s = w[m].sum()
            if s > 0 and m.sum() > 0:
                frac = w[m].max() / s
                if frac > worst[0]:
                    worst = (frac, o.key, int(m.sum()))
    print(f"{name:28s} {worst[1]:14s} {worst[0]*100:10.1f}% {worst[2]:6d}")
