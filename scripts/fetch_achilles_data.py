"""Populate ./achilles_data/ with the ACHILLES input tables ADoNIS reads, by
copying them out of the public oracle container image.  Run once after cloning:

    python scripts/fetch_achilles_data.py

Requires Docker.  The image is the single source of truth (see docs/CONTAINER.md);
alternatively set ACHILLES_DATA to a local ACHILLES `data/` directory and skip this.
"""
import os
import subprocess
import sys
from pathlib import Path

IMAGE = os.environ.get("ACHILLES_IMAGE", "ghcr.io/cesarjesusvalls/achilles:oracle")
ROOT = Path(__file__).resolve().parent.parent
DEST = Path(os.environ.get("ACHILLES_DATA", ROOT / "achilles_data"))
FILES = ["dcc_EW.dat", "Spectral_Functions/pke12p_tot.data"]


def main():
    if subprocess.run(["docker", "version"], capture_output=True).returncode != 0:
        sys.exit("Docker is required (or set ACHILLES_DATA to a local ACHILLES data dir).")
    print(f"pulling {IMAGE} ...", flush=True)
    subprocess.run(["docker", "pull", IMAGE], check=True)
    cid = subprocess.run(["docker", "create", IMAGE], check=True,
                         capture_output=True, text=True).stdout.strip()
    try:
        for rel in FILES:
            out = DEST / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["docker", "cp", f"{cid}:/achilles/data/{rel}", str(out)],
                           check=True)
            print(f"  {rel}  ->  {out}  ({out.stat().st_size:,} bytes)")
    finally:
        subprocess.run(["docker", "rm", cid], capture_output=True)
    print(f"done. ACHILLES_DATA defaults to {DEST}")


if __name__ == "__main__":
    main()
