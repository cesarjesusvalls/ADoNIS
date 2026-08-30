#!/usr/bin/env bash
# Extract the ACHILLES input tables ADoNIS needs from the public reference image.
#
#   scripts/fetch_achilles_data.sh [destination]      (default: ./achilles_data)
#
# Then point ADoNIS at them:  export ACHILLES_DATA=<destination>
set -euo pipefail

DEST="${1:-$PWD/achilles_data}"
IMAGE="${ADONIS_CONTAINER:-ghcr.io/cesarjesusvalls/achilles:oracle}"

if command -v docker >/dev/null 2>&1; then
    RUNTIME=docker
elif command -v apptainer >/dev/null 2>&1; then
    RUNTIME=apptainer
else
    echo "need docker or apptainer on PATH" >&2
    exit 1
fi

mkdir -p "$DEST"

# Copy the data and flux trees whole: ADoNIS reads some of these tables directly and the ACHILLES
# binary reads others (data/Rules.yml, data/Particles.yml, data/default/), and a selective list
# drifts out of step with what a run needs.
if [ "$RUNTIME" = docker ]; then
    docker pull "$IMAGE"
    cid=$(docker create "$IMAGE")
    trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
    docker cp "$cid:/achilles/data/." "$DEST/"
    mkdir -p "$DEST/flux"
    docker cp "$cid:/achilles/flux/." "$DEST/flux/"
else
    sif="$DEST/.achilles-oracle.sif"
    apptainer pull --force "$sif" "docker://$IMAGE"
    apptainer exec "$sif" tar -C /achilles/data -cf - . | tar -C "$DEST" -xf -
    mkdir -p "$DEST/flux"
    apptainer exec "$sif" tar -C /achilles/flux -cf - . | tar -C "$DEST/flux" -xf -
fi

# The MicroBooNE run cards read a flux table upstream does not ship; it is the YAML above in
# ACHILLES' own format.
python3 "$(dirname "$0")/microboone_flux_to_dat.py" \
    "$DEST/flux/microboone_flux_numu.yaml" "$DEST/flux/microboone_numu.dat"

missing=""
for f in dcc_EW.dat Rules.yml Particles.yml Spectral_Functions/pke12p_tot.data \
         MesonBaryonAmplitudes/ANL flux/T2K_nu.dat flux/minerva_numu_fhc.dat \
         flux/microboone_numu.dat default/OptionDefaults.yml; do
    [ -e "$DEST/$f" ] || missing="$missing $f"
done
if [ -n "$missing" ]; then
    echo "error: these required files were not written:$missing" >&2
    exit 1
fi

echo
echo "wrote $DEST"
find "$DEST" -type f ! -name '*.sif' | sed 's/^/  /'
echo
echo "export ACHILLES_DATA=$DEST"
