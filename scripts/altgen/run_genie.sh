#!/bin/bash
# Runs INSIDE the LUCiD container. Builds the T2K flux TH1, generates GENIE
# AR23_20i numu events on 12C with that flux, converts ghep -> gst.
# Env:
#   NEV    number of events (default 500000)
#   SEED   RNG seed (default 20240702)
#   EGLIST event-generator list (default CCQERES; custom lists in /genie_cfg)
#   NAME   output basename (default genie_t2k_12C_ar23_<EGLIST>)
# GXMLPATH must include /genie_cfg (custom generator lists + decayer override).
set -euo pipefail
NEV="${NEV:-500000}"
SEED="${SEED:-20240702}"
EGLIST="${EGLIST:-CCQERES}"
TUNE="AR23_20i_00_000"
TARGET=1000060120          # 12C ion PDG code
PROBE=14                   # numu
WORK=/work
NAME="${NAME:-genie_t2k_12C_ar23_${EGLIST}}"
export GXMLPATH=/genie_cfg:${GXMLPATH:-}

echo "=== [$(date -u +%H:%M:%S)] building flux histogram ==="
root -l -b -q "/scripts/make_flux_root.C(\"$WORK/T2K_nu.dat\",\"$WORK/t2k_flux.root\",\"t2kflux\")"

echo "=== [$(date -u +%H:%M:%S)] gevgen: $NEV numu on 12C, tune $TUNE, list $EGLIST ==="
cd "$WORK"
gevgen \
  -n "$NEV" \
  -p "$PROBE" \
  -t "$TARGET" \
  -e 0,30 \
  -f "$WORK/t2k_flux.root,t2kflux" \
  --cross-sections "$GENIE_XSEC_FILE" \
  --tune "$TUNE" \
  --seed "$SEED" \
  --event-generator-list "$EGLIST" \
  -o "$WORK/$NAME"

echo "=== [$(date -u +%H:%M:%S)] gevgen done; ghep file (GENIE writes exactly -o name): ==="
ls -la "$WORK/$NAME"

echo "=== [$(date -u +%H:%M:%S)] gntpc -> gst ==="
gntpc -i "$WORK/$NAME" -f gst -o "$WORK/${NAME}.gst.root"
echo "=== [$(date -u +%H:%M:%S)] DONE. gst: ==="
ls -la "$WORK/${NAME}.gst.root"
