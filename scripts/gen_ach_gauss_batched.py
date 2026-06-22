"""Build a HIGH-STAT Gaussian ACHILLES C reference by batching over distinct seeds (ACHILLES has a
sporadic deterministic mid-run SIGSEGV; a fixed seed always crashes at the same event, so we vary the
seed per batch and keep whatever each batch produced -- the partial hepmc stays valid).  Then extract +
concat each batch with the correct absolute normalization:
  combined per-event weight [nb] = w_i * weight_to_nb_b / B   (B = #batches) -> sum = mean batch sigma.
Usage: python -u scripts/gen_ach_gauss_batched.py [target_events=2000000] [batch=500000]
Progress: tqdm over accumulated events + per-batch lines.
"""
import os, sys, subprocess, glob
from pathlib import Path
import numpy as np
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 500_000
BASE = Path("_oracle_out/run_T2K_C_fate_gauss.yml")
BDIR = Path("data/oracle/_ach_gauss_batches"); BDIR.mkdir(parents=True, exist_ok=True)
OUT = "data/oracle/t2k_cc1pi_rich_ach_FSI_C_gauss.npz"
base_txt = BASE.read_text()

def card_for(seed, nev):
    t = base_txt.replace("  NEvents: 2000000", f"  NEvents: {nev}")
    t = t.replace("      Name: /out/T2K_C_fate_gauss.hepmc", f"      Name: /out/T2K_C_fate_gauss_s{seed}.hepmc")
    t = t.replace('Options: !include "data/default/OptionDefaults.yml"',
                  f"Options:\n  Initialize:\n    Seed: {seed}\n    Accuracy: 1e-2\n"
                  f"  Unweighting:\n    Name: Percentile\n    percentile: 99")
    p = Path(f"_oracle_out/run_T2K_C_fate_gauss_s{seed}.yml"); p.write_text(t); return p

def run_batch(seed):
    card = card_for(seed, BATCH)
    hep = Path(f"_oracle_out/T2K_C_fate_gauss_s{seed}.hepmc")
    log = f"/tmp/ach_gauss_s{seed}.log"
    print(f"  [batch seed={seed}] docker run (NEvents={BATCH}) -> {log}", flush=True)
    with open(log, "w") as fh:
        rc = subprocess.run(["docker", "run", "--rm", "-v", f"{os.getcwd()}/_oracle_out:/out",
                             "--entrypoint", "/achilles/bin/achilles", "achilles:fullcascade",
                             f"/out/run_T2K_C_fate_gauss_s{seed}.yml"], stdout=fh, stderr=subprocess.STDOUT).returncode
    if not hep.exists():
        print(f"  [batch seed={seed}] NO hepmc (rc={rc}) -- skip", flush=True); return None
    npz = BDIR / f"s{seed}.npz"
    subprocess.run([sys.executable, "-u", "scripts/extract_cc1pi_rich.py", str(hep), str(npz)], check=True)
    d = np.load(npz); n = len(d["w"]); print(f"  [batch seed={seed}] {n} events (rc={rc})", flush=True)
    return n

# --- accumulate batches until TARGET ---
total, seed = 0, 0
bar = tqdm(total=TARGET, desc="ACHILLES Gaussian events", file=sys.stdout, unit="ev")
while total < TARGET:
    seed += 1
    n = run_batch(seed)
    if n: total += n; bar.update(n)
    if seed > 60: print("  too many batches, stopping"); break
bar.close()

# --- combine batch npzs with correct normalization ---
files = sorted(glob.glob(str(BDIR / "s*.npz")))
B = len(files); print(f"\ncombining {B} batches, total ~{total} events", flush=True)
acc = {}
for f in files:
    d = np.load(f, allow_pickle=True)
    wnb = d["w"].astype(float) * float(d["weight_to_nb"]) / B           # combined absolute nb weight
    for k in d.files:
        if k in ("weight_to_nb", "gen_xs_pb", "sum_w_all"):
            continue
        acc.setdefault(k, []).append(d["w"]*0 + wnb if k == "w" else d[k])
out = {k: np.concatenate(v) for k, v in acc.items()}
out["weight_to_nb"] = 1.0                                               # w already in nb
np.savez(OUT, **out)
print(f"DONE -> {OUT}  ({len(out['w'])} events, weight_to_nb folded in)", flush=True)
