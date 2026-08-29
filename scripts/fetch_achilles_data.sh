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

FILES=(
    "dcc_EW.dat"
    "Spectral_Functions/pke12p_tot.data"
    "Spectral_Functions/pke12n_tot.data"
    "Spectral_Functions/pke40p_tot.data"
    "Spectral_Functions/pke40n_tot.data"
)

mkdir -p "$DEST/Spectral_Functions" "$DEST/MesonBaryonAmplitudes"

if [ "$RUNTIME" = docker ]; then
    docker pull "$IMAGE"
    cid=$(docker create "$IMAGE")
    trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
    for f in "${FILES[@]}"; do
        docker cp "$cid:/achilles/data/$f" "$DEST/$f"
    done
    docker cp "$cid:/achilles/data/MesonBaryonAmplitudes/ANL" "$DEST/MesonBaryonAmplitudes/ANL"
else
    sif="$DEST/.achilles-oracle.sif"
    apptainer pull --force "$sif" "docker://$IMAGE"
    for f in "${FILES[@]}"; do
        apptainer exec "$sif" cat "/achilles/data/$f" > "$DEST/$f"
    done
    apptainer exec "$sif" tar -C /achilles/data/MesonBaryonAmplitudes -cf - ANL | tar -C "$DEST/MesonBaryonAmplitudes" -xf -
fi

echo
echo "wrote $DEST"
find "$DEST" -type f ! -name '*.sif' | sed 's/^/  /'
echo
echo "export ACHILLES_DATA=$DEST"
