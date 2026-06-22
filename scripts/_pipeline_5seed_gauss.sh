#!/bin/bash
# Post-2M-ACHILLES pipeline: correct combine -> 5-seed Gaussian ADoNIS (RES, QE) -> CG matrix -> PDF.
# Runs AFTER the docker 2M batch finishes (sequential, no contention).  Each step streams to its log.
cd /Users/homelab/Lab/Playground/projects/DIFFGEN/ADoNIS || exit 1
PY=.venv/bin/python
say(){ echo "[$(date +%H:%M)] $*"; }

say "1/5 combine 2M Gaussian ACHILLES ref (correct pooled)"
$PY -u scripts/combine_ach_gauss.py > /tmp/combine_gauss.log 2>&1 || { say "combine FAIL"; exit 1; }
tail -1 /tmp/combine_gauss.log

say "2/5 ADoNIS 5-seed Gaussian RES (per-seed progress -> /tmp/gen_res_5seedG.log)"
$PY -u scripts/adonis_generate.py configs/gen_c_res_5seedG.yaml > /tmp/gen_res_5seedG.log 2>&1 || say "RES gen nonzero exit (per-seed checkpoint kept)"

say "3/5 ADoNIS 5-seed Gaussian QE (per-seed progress -> /tmp/gen_qe_5seedG.log)"
$PY -u scripts/adonis_generate.py configs/gen_c_qe_5seedG.yaml > /tmp/gen_qe_5seedG.log 2>&1 || say "QE gen nonzero exit (per-seed checkpoint kept)"

say "4/5 CG matrix (Gaussian C, 24 cells)"
rm -f paper_figures/matrix/*.png
$PY -u scripts/gen_cc_matrix.py CG fsi > /tmp/matrix_CG.log 2>&1 || say "matrix nonzero exit"
tail -3 /tmp/matrix_CG.log

say "5/5 build matrix PDF"
$PY -u scripts/build_matrix_pdf.py paper_figures/cc_matrix_CG.pdf paper_figures/matrix > /tmp/pdf_CG.log 2>&1 || say "pdf FAIL"
tail -1 /tmp/pdf_CG.log
say "PIPELINE_DONE -> paper_figures/cc_matrix_CG.pdf"
