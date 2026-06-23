"""Generate the ACHILLES half of the cascade-vertex matrix: run achilles:vertex (ACHILLES_VERTEXDUMP=1)
on the Gaussian T2K card and collect the per-primary `VTX ...` stderr lines into
cascade_vertex_<TAG>_ach.txt -- the input to scripts/cascade_vertex_matrix.py.  Mirrors the docker
invocation of gen_ach_gauss_batched.py (same base card / Unweighting), so it is apples-to-apples with the
cross-section-matrix banks and reproducible for any material.

Run: python -u scripts/gen_ach_vertex.py [TAG=C] [NEvents=500000] [seed=1] [out.txt]
"""
import os, sys, re, subprocess
from pathlib import Path

TAG = sys.argv[1] if len(sys.argv) > 1 else "C"
NEV = int(sys.argv[2]) if len(sys.argv) > 2 else 500_000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1
OUT = sys.argv[4] if len(sys.argv) > 4 else f"/tmp/cascade_vertex_{TAG}_ach.txt"
IMAGE = os.environ.get("ACH_VERTEX_IMAGE", "achilles:vertex")
STEM = f"T2K_{TAG}_fate_gauss"
BASE = Path(f"_oracle_out/run_{STEM}.yml")
base_txt = BASE.read_text()


def card_for(seed, nev):
    t = re.sub(r"NEvents:\s*\d+", f"NEvents: {nev}", base_txt, count=1)   # robust: no chained-prefix double-replace
    t = t.replace(f"      Name: /out/{STEM}.hepmc", f"      Name: /out/{STEM}_vtx_s{seed}.hepmc")
    t = t.replace('Options: !include "data/default/OptionDefaults.yml"',
                  f"Options:\n  Initialize:\n    Seed: {seed}\n    Accuracy: 1e-2\n"
                  f"  Unweighting:\n    Name: Percentile\n    percentile: 99")
    p = Path(f"_oracle_out/run_{STEM}_vtx_s{seed}.yml"); p.write_text(t); return p


def main():
    card = card_for(SEED, NEV)
    log = f"/tmp/ach_vertex_run_{TAG}_s{SEED}.log"
    print(f"[ach-vertex] {IMAGE} NEvents={NEV} seed={SEED} -> stderr {log}", flush=True)
    with open(log, "w") as fh:
        rc = subprocess.run(["docker", "run", "--rm", "-e", "ACHILLES_VERTEXDUMP=1",
                             "-v", f"{os.getcwd()}/_oracle_out:/out",
                             "--entrypoint", "/achilles/bin/achilles", IMAGE,
                             f"/out/run_{STEM}_vtx_s{SEED}.yml"],
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    # collect VTX lines (robust to the sporadic ACHILLES mid-run SIGSEGV: keep whatever was produced)
    n = 0
    with open(log) as fh, open(OUT, "w") as out:
        for line in fh:
            if line.startswith("VTX "):
                out.write(line); n += 1
    print(f"[ach-vertex] rc={rc}  wrote {n} VTX lines -> {OUT}", flush=True)
    if n == 0:
        print("[ach-vertex] WARNING: no VTX lines -- check the image has the VERTEXDUMP instrumentation", flush=True)


if __name__ == "__main__":
    main()
