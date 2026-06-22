#!/bin/bash
# Overnight: clean biased-sampler banks, regenerate pre-FSI + post-FSI ADoNIS banks (C, Ar) with the
# FIXED spectral importance sampler, then build the exclusive FSI + no-FSI matrices/PDFs.
# HARD design constraint: <=5h wall clock.  Soft gen-deadline 4.5h; plots run after.
# macOS has no `timeout`, so use a portable watchdog (background + sleep + kill).
cd /Users/homelab/Lab/Playground/projects/DIFFGEN/ADoNIS || exit 1
PY=.venv/bin/python
L=/tmp/overnight.log; : > "$L"
START=$(date +%s); DEADLINE=$((START + 16200))      # 4.5h soft deadline for generation
say(){ echo "[$(date +%H:%M) +$(( ($(date +%s)-START)/60 ))m] $*" | tee -a "$L"; }
left(){ echo $((DEADLINE - $(date +%s))); }

run_to(){  # $1=timeout_sec  rest=command -- portable timeout via watchdog
  local t=$1; shift
  "$@" & local pid=$!
  ( sleep "$t"; kill -9 "$pid" 2>/dev/null ) & local wd=$!
  wait "$pid" 2>/dev/null; local rc=$?
  kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
  return $rc
}

say "START overnight regen (fixed sampler).  soft gen-deadline 4.5h, hard cap 5h"

# 1. CLEAN target banks (regenerate fresh with the fixed sampler; incl old biased high-seed tags)
for b in t2k_cc1pi_engine_rich_cv5 t2k_cc0pi_engine_rich_cv5 \
         t2k_cc1pi_engine_rich_av5 t2k_cc0pi_engine_rich_av5 \
         t2k_cc1pi_engine_rich_cv12 t2k_cc0pi_engine_rich_cv30 \
         t2k_cc1pi_engine_rich_av12 t2k_cc0pi_engine_rich_av30 \
         t2k_cc1pi_engine_rich_nofsi t2k_cc0pi_engine_rich_nofsi \
         t2k_cc1pi_engine_rich_ar_nofsi t2k_cc0pi_engine_rich_ar_nofsi; do
  rm -f "data/oracle/$b.npz"
done
say "cleaned target banks (incl old biased cv12/cv30/av12/av30)"

gen(){  # $1=config  $2=timeout_sec  $3=label
  if [ "$(left)" -lt $(($2 + 300)) ]; then say "SKIP $3 (only $(left)s left before gen-deadline)"; return; fi
  say "GEN start $3 (budget $2s)"
  if run_to "$2" $PY -u scripts/adonis_generate.py "configs/$1.yaml" > "/tmp/gen_$1.log" 2>&1; then
    say "GEN done $3"
  else
    say "GEN cut/fail $3 (per-seed checkpoint -> partial bank valid)"
  fi
}

# 2. PRE-FSI (no cascade, fast)
gen gen_cc0pi_nofsi 1200 "preFSI C QE"
gen gen_cc1pi_nofsi 1200 "preFSI C RES"
gen gen_ar_nofsi    1800 "preFSI Ar QE+RES"

# 3. POST-FSI (cascade, sequential single-job for Ar OOM safety)
gen gen_c_qe_cv5   1500 "postFSI C QE"
gen gen_c_res_cv5  3000 "postFSI C RES"
gen gen_ar_qe_av5  2400 "postFSI Ar QE"
gen gen_ar_res_av5 3900 "postFSI Ar RES"

# 4. MATRICES + PDFs (FSI and no-FSI), each watchdogged
say "MATRIX FSI (post-FSI, 48 cells)"
run_to 2400 $PY -u scripts/gen_cc_matrix.py all fsi > /tmp/matrix_fsi.log 2>&1 && say "matrix FSI ok" || say "matrix FSI FAIL"
run_to 600  $PY -u scripts/build_matrix_pdf.py paper_figures/cc_matrix_FSI.pdf paper_figures/matrix > /tmp/pdf_fsi.log 2>&1 && say "pdf FSI ok" || say "pdf FSI FAIL"
say "MATRIX noFSI (pre-FSI)"
run_to 1800 $PY -u scripts/gen_cc_matrix.py all nofsi > /tmp/matrix_nofsi.log 2>&1 && say "matrix noFSI ok" || say "matrix noFSI FAIL"
run_to 600  $PY -u scripts/build_matrix_pdf.py paper_figures/cc_matrix_noFSI.pdf paper_figures/matrix_nofsi > /tmp/pdf_nofsi.log 2>&1 && say "pdf noFSI ok" || say "pdf noFSI FAIL"

say "PIPELINE COMPLETE  (total $(( ($(date +%s)-START)/60 ))m)"
