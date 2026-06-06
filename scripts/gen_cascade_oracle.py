"""Generate an ACHILLES cascade reaction-cross-section oracle (Fig c12_ar40).

Drives the cascade-enabled image (docker/Dockerfile.cascade -> achilles:cascade) over
several seeds, accumulating events (the cascade has a sporadic seed-specific mid-run
segfault; batching over seeds works around it), and saves the reaction sigma(p_pi) shape
(N_hits(p), since the beam is uniform in p) peak-normalised.

  python scripts/gen_cascade_oracle.py 40Ar virt 5     # nucleus, mode (virt|prop), nseeds

Requires `docker` + the `achilles:cascade` image and a self-contained base config in
`_oracle_out/cascade_<nucleus>_<mode>.yml` (see docs/phases/PHASE_G.md for the recipe).
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cascade_xsec_from_hepmc import events

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_oracle_out"
IMAGE = "achilles:cascade"


def run_batches(base_yml, nseeds, tag):
    """Run the base config over nseeds seeds, each writing its own hepmc; accumulate the
    incoming-pion momenta over all events."""
    import re
    P = []
    base = (OUT / base_yml).read_text()
    for seed in range(111, 111 + 222 * nseeds, 222):
        hep_name = f"{tag}_{seed}.hepmc"
        cfg = base
        cfg = re.sub(r"seed:\s*\d+", f"seed: {seed}", cfg)
        cfg = re.sub(r"Name: /out/\S+\.hepmc", f"Name: /out/{hep_name}", cfg)
        cfg_path = OUT / f"{tag}_{seed}.yml"
        cfg_path.write_text(cfg)
        subprocess.run(["docker", "run", "--rm", "-v", f"{OUT}:/out",
                        "--entrypoint", "/achilles/bin/achilles-cascade", IMAGE, f"/out/{cfg_path.name}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        hp = OUT / hep_name
        if hp.exists():
            P += [p for p, w in events(hp) if p is not None]
    return np.array(P)


def save_shape(P, csv, title):
    h, edges = np.histogram(P, bins=14, range=(80, 500))
    cen = 0.5 * (edges[:-1] + edges[1:])
    shape = h / h.max() if h.max() else h
    with open(ROOT / "data" / "oracle" / csv, "w") as f:
        f.write(f"# {title} ({len(P)} cascade events). Columns: p_pi[MeV], sigma_shape (peak-norm)\n")
        for c, s in zip(cen, shape):
            f.write(f"{c:.1f}  {s:.4f}\n")
    return cen[int(np.argmax(h))]


if __name__ == "__main__":
    nucleus = sys.argv[1] if len(sys.argv) > 1 else "12C"
    mode = sys.argv[2] if len(sys.argv) > 2 else "virt"
    nseeds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    base = {"12C": ("cascade_virt_c12.yml", "cascade_prop_c12.yml"),
            "40Ar": ("cascade_ar40.yml", "cascade_ar40.yml")}[nucleus][0 if mode == "virt" else 1]
    P = run_batches(base, nseeds, f"{nucleus}_{mode}")
    pk = save_shape(P, f"cascade_pip_{nucleus.lower()}_reaction_{mode}.csv",
                    f"ACHILLES cascade {mode} pi+ on {nucleus} reaction sigma")
    print(f"{nucleus} {mode}: {len(P)} events, Delta peak p_pi={pk:.0f} MeV")
</content>
