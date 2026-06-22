#!/bin/bash
# Ar all-Gaussian pipeline: batched Gaussian ACHILLES-Ar ref (own pooled combine) -> 5-seed Gaussian
# ADoNIS Ar (RES, QE) -> ArG matrix -> PDF.  Sequential (docker then jax -> no contention).
cd /Users/homelab/Lab/Playground/projects/DIFFGEN/ADoNIS || exit 1
PY=.venv/bin/python
say(){ echo "[$(date +%H:%M)] $*"; }

say "1/5 ACHILLES Ar Gaussian 2M (batched, seed-varied, pooled combine -> t2k_cc1pi_rich_ach_FSI_Ar_gauss.npz)"
$PY -u scripts/gen_ach_gauss_batched.py 2000000 500000 Ar > /tmp/ach_gauss_Ar_batched.log 2>&1 || say "ACH Ar gen nonzero exit"
tail -2 /tmp/ach_gauss_Ar_batched.log

say "2/5 ADoNIS 5-seed Gaussian Ar RES (-> /tmp/gen_res_ar5seedG.log)"
$PY -u scripts/adonis_generate.py configs/gen_ar_res_5seedG.yaml > /tmp/gen_res_ar5seedG.log 2>&1 || say "Ar RES gen nonzero exit (per-seed checkpoint kept)"

say "3/5 ADoNIS 5-seed Gaussian Ar QE (-> /tmp/gen_qe_ar5seedG.log)"
$PY -u scripts/adonis_generate.py configs/gen_ar_qe_5seedG.yaml > /tmp/gen_qe_ar5seedG.log 2>&1 || say "Ar QE gen nonzero exit (per-seed checkpoint kept)"

say "4/5 ArG matrix (Gaussian Ar, 24 cells)"
rm -f paper_figures/matrix/*.png
$PY -u scripts/gen_cc_matrix.py ArG fsi > /tmp/matrix_ArG.log 2>&1 || say "matrix nonzero exit"
tail -3 /tmp/matrix_ArG.log

say "5/5 build matrix PDF"
$PY -u scripts/build_matrix_pdf.py paper_figures/cc_matrix_ArG.pdf paper_figures/matrix > /tmp/pdf_ArG.log 2>&1 || say "pdf FAIL"
tail -1 /tmp/pdf_ArG.log
say "PIPELINE_DONE -> paper_figures/cc_matrix_ArG.pdf"
