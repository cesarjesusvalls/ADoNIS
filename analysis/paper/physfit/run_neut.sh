#!/bin/bash
# Runs INSIDE the LUCiD container. NEUT numu on pure 12C with the T2K ND280 flux
# (same t2k_flux.root the GENIE runs use), then NUISANCE nuisflat -> GenericVectors flat tree.
# Card derived from the stock $NEUT_CARDS/neut_5.4.0_1GeV_C.card (pure 12C: 6n+6p, 0 free p),
# with the fixed-1GeV energy replaced by the flux histogram.
#   NEV  : number of events (default 300000)
set -euo pipefail
NEV="${NEV:-300000}"
WORK=/work
NAME="${NAME:-neut_t2k_12C}"
CARD=$WORK/${NAME}.card

cp "$NEUT_CARDS/neut_5.4.0_1GeV_C.card" "$CARD"
# events + flux-histogram energy sampling (GeV units)
sed -i "s/^EVCT-NEVT .*/EVCT-NEVT  $NEV/" "$CARD"
sed -i "s/^EVCT-MPV .*/EVCT-MPV  3/" "$CARD"
sed -i "s|^EVCT-PV .*|EVCT-FILENM '$WORK/t2k_flux.root'\nEVCT-HISTNM 't2kflux'\nEVCT-INMEV 0|" "$CARD"
# stock card uses a relative crsdat path; resolve to the container install (env NEUT_CRSPATH)
sed -i "s|^NEUT-CRSPATH .*|NEUT-CRSPATH '$NEUT_CRSPATH'|" "$CARD"
echo "=== card diff vs stock (non-comment) ==="
grep -vE "^C|^$" "$CARD"

echo "=== [$(date -u +%H:%M:%S)] neutroot2: $NEV numu on 12C ==="
cd "$WORK"
neutroot2 "$CARD" "$WORK/${NAME}.root"
echo "=== [$(date -u +%H:%M:%S)] PrepareNEUT (flux/xsec norm) ==="
PrepareNEUT -i "$WORK/${NAME}.root" -f "$WORK/t2k_flux.root,t2kflux" -G
echo "=== [$(date -u +%H:%M:%S)] nuisflat -> GenericVectors ==="
nuisflat -i "NEUT:$WORK/${NAME}.root" -f GenericVectors -o "$WORK/${NAME}_flat.root"
echo "=== [$(date -u +%H:%M:%S)] DONE ==="
ls -la "$WORK/${NAME}"*
