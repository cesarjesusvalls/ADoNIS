#!/usr/bin/env bash
# Build an ACHILLES reference image.
#
#     docker/build.sh cascade|fullcascade <path-to-achilles-checkout> [output.sif]
#
# cascade      standalone achilles-cascade: hadron-nucleus cross sections.
# fullcascade  the same fix linked into the main achilles binary, for in-event FSI.
#
# The recipe reads its source from a directory named Achilles-src beside itself, so the checkout is
# staged there first; nothing depends on where the checkout actually lives.
set -euo pipefail

IMAGE="${1:?usage: build.sh cascade|fullcascade <achilles-checkout> [out.sif]}"
SRC="${2:?usage: build.sh cascade|fullcascade <achilles-checkout> [out.sif]}"
OUT="${3:-$IMAGE.sif}"

case "$IMAGE" in
    cascade|fullcascade) ;;
    *) echo "unknown image '$IMAGE' (expected cascade or fullcascade)" >&2; exit 1 ;;
esac
[ -d "$SRC/src/Achilles" ] || { echo "'$SRC' is not an ACHILLES checkout (no src/Achilles)" >&2; exit 1; }

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

cp -a "$SRC" "$STAGE/Achilles-src"
cp "$HERE/patch_cascade_registry.py" "$STAGE/"
cp "$HERE/$IMAGE.def" "$STAGE/"

if command -v apptainer >/dev/null 2>&1; then
    ( cd "$STAGE" && apptainer build --fakeroot "$IMAGE.sif" "$IMAGE.def" )
    mv "$STAGE/$IMAGE.sif" "$OUT"
elif command -v docker >/dev/null 2>&1; then
    docker build -f "$HERE/Dockerfile.$IMAGE" -t "achilles-$IMAGE" "$STAGE/Achilles-src"
    echo "built docker image achilles-$IMAGE"
    exit 0
else
    echo "need apptainer or docker on PATH" >&2
    exit 1
fi

echo "wrote $OUT"
