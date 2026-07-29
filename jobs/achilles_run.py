#!/usr/bin/env python3
"""Run an ACHILLES run card via Apptainer on S3DF (the no-Docker replacement for
analysis/utils/run_achilles.py's `docker run`).

Reuses the tested image/binary selection (`_image_for`) and output-name parsing (`_output_name`) from
adonis.oracle.run_achilles, and maps each docker image tag to its local .sif:

    :oracle       -> achilles-oracle.sif        (no-cascade: inclusive, free-nucleon, res1pi, *_nofsi, T2K_H)
    :fullcascade  -> achilles-fullcascade.sif    (in-event QE+RES+FSI: run_T2K_*_fsi)
    :cascade      -> achilles-cascade.sif         (standalone pi/p/n cascade: run_cascade_*)

Per invocation it writes a card copy into <outdir> with NEvents + Seed overridden and a unique output
hepmc name (so seeds/shards don't clobber), then runs:

    apptainer exec --fakeroot --writable-tmpfs --pwd /achilles --bind <outdir>:/out <sif> <binary> /out/<card>

Usage:
    python jobs/achilles_run.py configs/achilles/run_T2K_H.yml --nevents 500 --seed 0 --out $ADONIS_OUT/achilles/T2K_H
    python jobs/achilles_run.py configs/achilles/run_cascade_pip_C_broad.yml --nevents 500000 --seed 3 --out .../beam_pip
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path("/sdf/home/c/cjesus/DIFFGEN/ADoNIS")
sys.path.insert(0, str(REPO))
from adonis.oracle.run_achilles import _image_for, _output_name  # tested logic (P4: analysis.utils -> adonis.oracle)

IMG = "/sdf/data/neutrino/cjesus/software/images"
SIF = {
    "ghcr.io/cesarjesusvalls/achilles:oracle": f"{IMG}/achilles-oracle.sif",
    "achilles:fullcascade": f"{IMG}/achilles-fullcascade.sif",
    "achilles:cascade": f"{IMG}/achilles-cascade.sif",
}


def _seed_the_card(raw, seed):
    """Set the ACHILLES RNG seed where each binary ACTUALLY reads it (verified against the fork source):
      * EventGen (neutrino gen: achilles):  Options/Initialize/Seed   (EventGen.cc)
      * RunCascade (achilles-cascade):      top-level Initialize/seed  (RunCascade.cc)
    The neutrino cards set Options via `!include data/default/OptionDefaults.yml`, whose Seed defaults to
    a FIXED 12345678 -> every shard was identical. We inline that default with Seed overridden, and also
    add a top-level Initialize/seed for the cascade binary. (A `Main: Seed` -- what we set before -- is
    read by NEITHER, so all shards were byte-identical: the 10M oracle was 200k events x N copies.)"""
    data = os.environ.get("ACHILLES_DATA", "/sdf/data/neutrino/cjesus/ADoNIS/software/achilles_data")
    m = re.search(r'^Options:\s*!include\s*"?([^"\n]+)"?\s*$', raw, re.M)
    if m:
        inc_rel = m.group(1).strip()
        inc_host = Path(data) / (inc_rel.split("data/", 1)[-1] if inc_rel.startswith("data/") else inc_rel)
        opts = yaml.safe_load(Path(inc_host).read_text()) or {}
        opts.setdefault("Initialize", {})["Seed"] = int(seed)
        dump = yaml.safe_dump(opts, default_flow_style=False, sort_keys=False)
        indented = "".join("  " + ln + "\n" for ln in dump.splitlines())
        raw = raw[:m.start()] + "Options:\n" + indented + raw[m.end():]
    # top-level Initialize/seed for RunCascade (and harmless for EventGen, which reads Options/...).
    # The cascade cards ALREADY have `Initialize:\n  seed: <n>` — REPLACE the value (inserting a second
    # `seed:` key makes yaml-cpp abort with NonUniqueMapKey).
    m2 = re.search(r'^Initialize:[ \t]*\n((?:[ \t]+\S.*\n)*)', raw, re.M)
    if m2:
        block = m2.group(0)
        nblock, nsub = re.subn(r'^([ \t]+)(seed|Seed):.*$', rf'\g<1>\g<2>: {seed}', block, flags=re.M)
        if not nsub:
            nblock = block.replace("Initialize:", f"Initialize:\n  seed: {seed}", 1)
        raw = raw[:m2.start()] + nblock + raw[m2.end():]
    else:
        raw += f"\nInitialize:\n  seed: {seed}\n  Seed: {seed}\n"
    return raw


def write_card(card_path, outdir, nevents, seed):
    """Copy the card into <outdir>/<stem>_s<seed>.yml with NEvents + real per-shard seed + unique hepmc."""
    raw = Path(card_path).read_text()
    base_hepmc = _output_name(raw)                        # e.g. /out/T2K_H.hepmc
    stem = Path(base_hepmc).name[:-len(".hepmc")]
    new_hepmc = f"/out/{stem}_s{seed}.hepmc"
    raw = raw.replace(base_hepmc, new_hepmc)
    if nevents is not None:
        raw = re.sub(r"NEvents:\s*\d+", f"NEvents: {nevents}", raw)
    raw = _seed_the_card(raw, seed)
    card_out = Path(outdir) / f"{Path(card_path).stem}_s{seed}.yml"
    card_out.write_text(raw)
    return card_out.name, Path(new_hepmc).name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("card")
    ap.add_argument("--nevents", type=int, default=None, help="override NEvents (default: card value)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output dir (mounted as /out)")
    ap.add_argument("--extract", help="run adonis.oracle.extract <channel> on the hepmc -> .npz")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    image, _native, binary = _image_for(Path(a.card).name)
    sif = SIF[image]
    if not Path(sif).exists():
        sys.exit(f"image not built yet: {sif}  (card {Path(a.card).name} needs {image})")
    os.makedirs(a.out, exist_ok=True)
    card_name, hepmc = write_card(a.card, a.out, a.nevents, a.seed)
    # Bind the extracted ACHILLES data/flux over /achilles/{data,flux} so every image (the Debian
    # oracle AND the alpine-built cascade images) reads the identical single-source-of-truth inputs.
    # NOTE: no --fakeroot — the alpine cascade images lack the `faked` daemon and --fakeroot breaks
    # exec there; --writable-tmpfs alone gives ACHILLES a writable /achilles for its log/References.
    data = os.environ.get("ACHILLES_DATA", "/sdf/data/neutrino/cjesus/ADoNIS/software/achilles_data")
    flux = os.environ.get("ACHILLES_FLUX", "/sdf/data/neutrino/cjesus/ADoNIS/software/achilles_flux")
    # ACHILLES writes achilles.log + References.txt to its CWD (/achilles, root-owned in the image).
    # Without --fakeroot (which the alpine cascade images can't use) those writes are permission-denied
    # intermittently. Bind WRITABLE host files over those two paths so the writes land in <out> instead.
    alog = Path(a.out) / "achilles.log"; arefs = Path(a.out) / "References.txt"
    alog.touch(exist_ok=True); arefs.touch(exist_ok=True)
    host_bin = os.environ.get("ACHILLES_HOST_BIN")
    run_env = dict(os.environ); cwd = None
    if host_bin:
        # HOST-BINARY mode (rebuilt-from-source achilles; fixes the oracle-image (e,e')+cascade SIGSEGV).
        # No container: rewrite the card's /out/ paths to the real out dir and run from a workdir whose
        # data/flux symlink to ACHILLES_DATA/FLUX (identical inputs), with ACH_LD on LD_LIBRARY_PATH.
        cf = Path(a.out) / card_name
        cf.write_text(cf.read_text().replace("/out/", str(Path(a.out).resolve()) + "/"))
        wd = Path(a.out) / "_wd"; wd.mkdir(exist_ok=True)
        for nm, tgt in (("data", data), ("flux", flux)):
            lk = wd / nm
            if not lk.exists():
                lk.symlink_to(tgt)
        # the binary reads top-level config files (FormFactors.yml, ...) from its CWD -> symlink them from
        # the source tree so a per-shard _wd is self-contained (and parallel shards don't clobber each other).
        src = Path(os.environ.get("ACHILLES_SRC", "/sdf/data/neutrino/cjesus/ADoNIS/software/Achilles-src"))
        for p in src.glob("*.yml"):
            lk = wd / p.name
            if not lk.exists():
                lk.symlink_to(p)
        run_env["LD_LIBRARY_PATH"] = os.environ.get("ACH_LD", "") + ":" + run_env.get("LD_LIBRARY_PATH", "")
        cmd = [host_bin, str(cf.resolve())]; cwd = str(wd)
    else:
        cmd = ["apptainer", "exec", "--writable-tmpfs", "--pwd", "/achilles",
               "--bind", f"{Path(a.out).resolve()}:/out",
               "--bind", f"{data}:/achilles/data", "--bind", f"{flux}:/achilles/flux",
               "--bind", f"{alog}:/achilles/achilles.log", "--bind", f"{arefs}:/achilles/References.txt",
               sif, binary, f"/out/{card_name}"]
    print("  $", " ".join(cmd), flush=True)
    if a.dry_run:
        return
    log = Path(a.out) / f"{Path(card_name).stem}.log"
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd, env=run_env).returncode
    hp = Path(a.out) / hepmc
    ok = hp.exists()
    print(f"{'OK' if ok else 'FAIL'} rc={rc} -> {hp}  (log {log})", flush=True)
    if ok and a.extract:
        npz = str(hp).replace(".hepmc", ".npz")
        r = subprocess.run([sys.executable, "-m", "adonis.oracle.extract", a.extract, str(hp), npz],
                           cwd=str(REPO))
        print(f"extract[{a.extract}] rc={r.returncode} -> {npz}", flush=True)
        ok = ok and r.returncode == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
