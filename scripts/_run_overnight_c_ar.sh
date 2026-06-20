#!/bin/zsh
# Overnight C+Ar generation (resonance RES + Pauli-fix + pool, all defaults) sized to <=6h, then
# extract the Ar proc-tagged ACHILLES reference and render the C & Ar CC0pi/CC1pi plots vs ACHILLES.
cd /Users/homelab/Lab/Playground/projects/DIFFGEN/ADoNIS
LOG=/tmp/overnight_c_ar.log; : > "$LOG"
echo "START $(date '+%F %T')" >> "$LOG"

gen() {  # $1 = config name (no path/ext)
  echo "=== GEN $1 ($(date +%T)) ===" >> "$LOG"
  JAX_ENABLE_X64=1 .venv/bin/python -u scripts/adonis_generate.py "configs/$1.yaml" >> "$LOG" 2>&1 \
    && echo "GEN OK $1 ($(date +%T))" >> "$LOG" || echo "GEN FAIL $1 ($(date +%T))" >> "$LOG"
}

# --- generation phase, HARD 6h backstop (per-seed checkpointed, so a kill keeps finished seeds) ---
( gen gen_c6h_qe; gen gen_ar6h_qe; gen gen_c6h_res; gen gen_ar6h_res ) &
GPID=$!
( sleep 21600; kill -TERM $GPID 2>/dev/null; echo "BUDGET 6h HIT -> stopped generation $(date +%T)" >> "$LOG" ) &
WPID=$!
wait $GPID 2>/dev/null
kill $WPID 2>/dev/null
echo "GEN PHASE DONE $(date '+%F %T')" >> "$LOG"

# --- Ar proc-tagged ACHILLES reference (needed for ref_proc QE/RES split on Ar) ---
if [ ! -f data/oracle/t2k_cc1pi_rich_ach_FSI_Ar_proc.npz ]; then
  echo "=== EXTRACT Ar proc reference ($(date +%T)) ===" >> "$LOG"
  JAX_ENABLE_X64=1 .venv/bin/python -u scripts/extract_cc1pi_rich.py \
    _oracle_out/T2K_Ar_virt.hepmc data/oracle/t2k_cc1pi_rich_ach_FSI_Ar_proc.npz >> "$LOG" 2>&1 \
    && echo "EXTRACT OK ($(date +%T))" >> "$LOG" || echo "EXTRACT FAIL ($(date +%T))" >> "$LOG"
fi

ana() {
  echo "=== ANA $1 ($(date +%T)) ===" >> "$LOG"
  JAX_ENABLE_X64=1 .venv/bin/python -u scripts/adonis_analyze.py "configs/$1.yaml" >> "$LOG" 2>&1 \
    && echo "ANA OK $1" >> "$LOG" || echo "ANA FAIL $1" >> "$LOG"
}
for c in ana_cc0pi_qeonly_c6h ana_cc0pi_c6h ana_cc1pi_resonly_c6h ana_cc1pi_c6h \
         ana_cc0pi_qeonly_ar6h ana_cc0pi_ar6h ana_cc1pi_resonly_ar6h ana_cc1pi_ar6h; do ana "$c"; done
echo "OVERNIGHT DONE $(date '+%F %T')" >> "$LOG"
