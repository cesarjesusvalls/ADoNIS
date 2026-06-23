#!/bin/zsh
# Worker-count scan: run K parallel cascade processes, each on N_PER events at fixed P, simultaneously.
# Aggregate throughput = (K * N_PER) / wall.  If per-process ev/s holds as K grows, a single kernel was
# underutilizing the cores (more workers help); if aggregate ev/s plateaus, one kernel already saturates.
# Each worker prints its own pure-run ev/s (compile excluded) via _engine_timing.py.
#
# Usage: scripts/_engine_workers.sh [N_PER] [P] [Klist e.g. "1 2 4 8"]
cd "$(dirname "$0")/.."
N_PER=${1:-8000}
P=${2:-16}
KLIST=${3:-"1 2 4 8"}
PY=.venv/bin/python
echo "[workers] N_PER=$N_PER P=$P cores=$(sysctl -n hw.ncpu)  Klist=$KLIST"
for K in ${=KLIST}; do
  t0=$(date +%s.%N)
  pids=()
  for w in $(seq 1 $K); do
    $PY -u scripts/_engine_timing.py $N_PER $P > /tmp/_worker_${K}_${w}.log 2>&1 &
    pids+=$!
  done
  for p in $pids; do wait $p; done
  t1=$(date +%s.%N)
  wall=$(echo "$t1 - $t0" | bc)
  # sum the per-worker pure-run ev/s (last data row of each worker log)
  evps_sum=$(for w in $(seq 1 $K); do tail -1 /tmp/_worker_${K}_${w}.log | awk '{print $4}'; done | paste -sd+ - | bc 2>/dev/null)
  echo "K=$K  wall=${wall}s  sum(per-worker run ev/s)=${evps_sum}  (aggregate throughput incl. compile = $(echo "$K * $N_PER / $wall" | bc 2>/dev/null) ev/s)"
done
