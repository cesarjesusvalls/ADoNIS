#!/bin/zsh
# Wait for the RES _cfix bank, then render the full plot set vs ACHILLES.
cd /Users/homelab/Lab/Playground/projects/DIFFGEN/ADoNIS
LOG=/tmp/cfix_plots.log
: > "$LOG"
echo "WAIT for RES bank $(date +%H:%M:%S)" >> "$LOG"
until grep -q "RES DONE" /tmp/gen_cfix_seq.log 2>/dev/null; do sleep 20; done
echo "RES bank ready $(date +%H:%M:%S)" >> "$LOG"
run() {
  echo "=== $1 ($(date +%H:%M:%S)) ===" >> "$LOG"
  JAX_ENABLE_X64=1 .venv/bin/python -u scripts/adonis_analyze.py "$1" >> "$LOG" 2>&1 \
    && echo "OK $1" >> "$LOG" || echo "FAIL $1" >> "$LOG"
}
run configs/ana_cc1pi_resonly_cfix.yaml
run configs/ana_cc1pi_cfix.yaml
run configs/ana_cc0pi_qeonly_cfix.yaml
run configs/ana_cc0pi_cfix.yaml
run configs/ana_cc0pi_0p_cfix.yaml
run configs/ana_cc0pi_1p_cfix.yaml
run configs/ana_cc0pi_2p_cfix.yaml
echo "ALL PLOTS DONE $(date +%H:%M:%S)" >> "$LOG"
