#!/bin/zsh
cd /Users/homelab/Lab/Playground/projects/DIFFGEN/ADoNIS
LOG=/tmp/res7_validate.log; : > "$LOG"
echo "WAIT for RES7 gen $(date +%H:%M:%S)" >> "$LOG"
until grep -q "RES7 GEN DONE" /tmp/gen_res7.log 2>/dev/null; do sleep 20; done
echo "RES7 bank ready $(date +%H:%M:%S)" >> "$LOG"
run() {
  echo "=== $1 ($(date +%H:%M:%S)) ===" >> "$LOG"
  JAX_ENABLE_X64=1 .venv/bin/python -u scripts/adonis_analyze.py "$1" >> "$LOG" 2>&1 \
    && echo "OK $1" >> "$LOG" || echo "FAIL $1" >> "$LOG"
}
run configs/ana_cc1pi_resonly_res7.yaml
run configs/ana_cc1pi_res7.yaml
# N_eff comparison: old tchannel cfix bank vs new resonance res7 bank (selected CC1pi signal)
echo "=== N_eff (selected CC1pi RES-only) old cfix vs new res7 ===" >> "$LOG"
JAX_ENABLE_X64=1 .venv/bin/python - >> "$LOG" 2>&1 <<'PY'
import numpy as np
from adonis.workflow.config import load_analysis_config
import adonis.workflow.signal as SG
def neff(w): w=np.asarray(w); return w.sum()**2/np.sum(w**2)
for tag,path in [("tchannel cfix","data/oracle/t2k_cc1pi_cfix.npz"),
                 ("resonance res7","data/oracle/t2k_cc1pi_res7.npz")]:
    cfg=load_analysis_config("configs/ana_cc1pi_resonly_res7.yaml")
    s=SG.select_signal(path, cfg.signal, carbon_only=cfg.carbon_only)
    print(f"  {tag:16s}: raw_bank? sel_count={len(s['w'])}  sel_Neff={neff(s['w']):.0f}  sigma={s['w'].sum():.4e}")
PY
echo "RES7 VALIDATE DONE $(date +%H:%M:%S)" >> "$LOG"
