#!/bin/zsh
cd /Users/homelab/Lab/Playground/projects/DIFFGEN/ADoNIS
LOG=/tmp/av5.log; : > "$LOG"
echo "START $(date '+%F %T')" >> "$LOG"
gen() {
  echo "=== GEN $1 ($(date +%T)) ===" >> "$LOG"
  JAX_ENABLE_X64=1 .venv/bin/python -u scripts/adonis_generate.py "configs/$1.yaml" >> "$LOG" 2>&1 \
    && echo "GEN OK $1 ($(date +%T))" >> "$LOG" || { echo "GEN FAIL $1" >> "$LOG"; exit 1; }
}
ana() {
  echo "=== ANA $1 ($(date +%T)) ===" >> "$LOG"
  JAX_ENABLE_X64=1 .venv/bin/python -u scripts/adonis_analyze.py "configs/$1.yaml" >> "$LOG" 2>&1 \
    && echo "ANA OK $1" >> "$LOG" || echo "ANA FAIL $1" >> "$LOG"
}
gen gen_ar_res_av5      # builds the larger material-keyed grid res_vegasgrid_Ar40.npz, then 5 seeds
gen gen_ar_qe_av5       # 5 seeds, no grid
for c in ana_cc0pi_qeonly_av5 ana_cc0pi_av5 ana_cc0pi_0p_av5 ana_cc0pi_1p_av5 ana_cc0pi_2p_av5 \
         ana_cc1pi_resonly_av5 ana_cc1pi_av5; do ana "$c"; done
echo "AV5 ALL DONE $(date '+%F %T')" >> "$LOG"
