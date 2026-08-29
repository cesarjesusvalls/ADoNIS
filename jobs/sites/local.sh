# No batch system: the repository itself, whatever python is on PATH.
ADONIS_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ADONIS_DATA=${ADONIS_DATA:-$ADONIS_REPO}

ADONIS_PY_GPU=${ADONIS_PY_GPU:-$(command -v python3)}
ADONIS_PY_CPU=${ADONIS_PY_CPU:-$(command -v python3)}

ACHILLES_DATA=${ACHILLES_DATA:-$ADONIS_REPO/achilles_data}
