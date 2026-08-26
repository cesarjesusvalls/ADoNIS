"""Shim kept so the S3DF submitters keep working; the tool is analysis.oracle_tools.run_achilles,
which now selects the container runtime and the image itself.

    python jobs/achilles_run.py <card> --nevents N --seed S --out DIR [--extract CHANNEL]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.oracle_tools.run_achilles import main

if __name__ == "__main__":
    argv = []
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--seed":
            argv += ["--seed0", next(it)]
        elif a == "--out":
            argv += ["--out-dir", next(it)]
        else:
            argv.append(a)
    main(argv)
