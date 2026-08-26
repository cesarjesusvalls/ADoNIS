"""Run an ACHILLES run card in its container: configs/achilles/<card>.yml -> output/achilles/<...>.hepmc.

Three images cover the cards, selected by card prefix: `cascade` for the standalone hadron cascade,
`fullcascade` for in-event QE+RES+FSI, and `oracle` for everything without a cascade.  Under docker
the oracle image is amd64 and needs --platform on an arm64 host; the two cascade images must run
native, since amd64 emulation faults in the cascade.

The runtime is docker or apptainer, whichever ADONIS_CONTAINER names, else whichever is on PATH.
Apptainer resolves each image to <ACHILLES_IMAGES>/achilles-<tag>.sif and binds the ACHILLES data and
flux trees, which docker carries inside the image.

--scan-energy overrides the beam energy; --scan-kick the pion kick momentum; --nevents the event
count.  ACHILLES can fault mid-run, so each seed's partial output is kept.

  python -m analysis.oracle_tools.run_achilles configs/achilles/run_inclusive_ee_C_qe.yml
  python -m analysis.oracle_tools.run_achilles configs/achilles/run_freenucleon_res_H.yml --scan-energy 500,1000,2000
  python -m analysis.oracle_tools.run_achilles configs/achilles/run_cascade_pip_C.yml --scan-kick 100,200,500
  python -m analysis.oracle_tools.run_achilles configs/achilles/run_T2K_C_fsi_gauss.yml --seeds 4 --extract cc1pi_rich
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from adonis.io import achilles_data_root


OUT_DIR = "output/achilles"
ORACLE = "ghcr.io/cesarjesusvalls/achilles:oracle"

_RULES = [
    ("run_cascade", ("achilles:cascade", True, "/achilles/bin/achilles-cascade")),
    ("run_T2K_C_fsi",   ("achilles:fullcascade", True, "/achilles/bin/achilles")),
    ("run_T2K_Ar_fsi",  ("achilles:fullcascade", True, "/achilles/bin/achilles")),
    ("run_MINERvA_C_fsi",    ("achilles:fullcascade", True, "/achilles/bin/achilles")),
    ("run_MicroBooNE_Ar_fsi", ("achilles:fullcascade", True, "/achilles/bin/achilles")),
    ("run_MicroBooNE_C_fsi", ("achilles:fullcascade", True, "/achilles/bin/achilles")),
]
_DEFAULT = (ORACLE, False, "/achilles/bin/achilles")


def _image_for(card_name):
    for prefix, rule in _RULES:
        if card_name.startswith(prefix):
            return rule
    return _DEFAULT


def container_runtime():
    """'docker' or 'apptainer': $ADONIS_CONTAINER if set, else whichever is installed."""
    want = os.environ.get("ADONIS_CONTAINER")
    if want:
        if want not in ("docker", "apptainer"):
            raise SystemExit(f"ADONIS_CONTAINER={want!r}: expected docker or apptainer")
        return want
    for name in ("docker", "apptainer"):
        if shutil.which(name):
            return name
    raise SystemExit("no container runtime found: install docker or apptainer, "
                     "or set ADONIS_CONTAINER to the one you have")


def image_file(image):
    """The .sif an image tag maps to, under $ACHILLES_IMAGES."""
    root = Path(os.environ.get("ACHILLES_IMAGES", "images"))
    return root / f"achilles-{image.rsplit(':', 1)[-1]}.sif"


def _docker_cmd(card_in_out, image, native, entrypoint, out_abs):
    cmd = ["docker", "run", "--rm"]
    if not native:
        cmd += ["--platform", "linux/amd64"]
    cmd += ["-v", f"{out_abs}:/out", "--entrypoint", entrypoint, image, f"/out/{card_in_out}"]
    return cmd


def _apptainer_cmd(card_in_out, image, _native, entrypoint, out_abs):
    """The image carries the binaries; the data and flux trees are bound from the host."""
    sif = image_file(image)
    if not sif.exists():
        raise SystemExit(f"image not found: {sif}\n"
                         f"  set ACHILLES_IMAGES to the directory holding achilles-*.sif, or build it "
                         f"from docker/{image.rsplit(':', 1)[-1]}.def")
    data = achilles_data_root()
    flux = Path(os.environ.get("ACHILLES_FLUX", str(data.parent / "achilles_flux")))
    # ACHILLES writes these two beside its working directory, which is read-only in the image.
    out = Path(out_abs)
    binds = [f"{out}:/out", f"{data}:/achilles/data", f"{flux}:/achilles/flux"]
    for name in ("achilles.log", "References.txt"):
        p = out / name
        p.touch(exist_ok=True)
        binds.append(f"{p}:/achilles/{name}")
    cmd = ["apptainer", "exec", "--writable-tmpfs", "--pwd", "/achilles"]
    for b in binds:
        cmd += ["--bind", b]
    return cmd + [str(sif), entrypoint, f"/out/{card_in_out}"]


_RUNTIME_CMD = {"docker": _docker_cmd, "apptainer": _apptainer_cmd}


def _write_card(card_path, out_dir, suffix, overrides):
    """Copy the run card into out_dir (the /out mount) with optional scan overrides.  The output hepmc keeps
    the card's ORIGINAL Output Name (so consumers find e.g. inclusive_ee_C_qe.hepmc), with `suffix` appended
    only for scan points / seeds so they don't clobber each other."""
    raw = Path(card_path).read_text()
    base_hepmc = _output_name(raw)
    hepmc_stem = Path(base_hepmc).name[: -len(".hepmc")]
    new_hepmc = f"/out/{hepmc_stem}{suffix}.hepmc"
    raw = raw.replace(base_hepmc, new_hepmc)
    overrides = dict(overrides)
    seed = overrides.pop("seed", None)
    nevents = overrides.pop("nevents", None)
    for key, val in overrides.items():
        raw = _override(raw, key, val)
    if nevents is not None:
        raw = re.sub(r"NEvents:\s*\d+", f"NEvents: {int(nevents)}", raw)
    if seed is not None:
        raw = seed_card(raw, int(seed))
    card_name = Path(card_path).stem + suffix + ".yml"
    (Path(out_dir) / card_name).write_text(raw)
    return card_name, new_hepmc.split("/")[-1]


def _output_name(raw):
    for ln in raw.splitlines():
        s = ln.strip()
        if s.startswith("Name:") and ".hepmc" in s:
            return s.split("Name:", 1)[1].strip()
    raise ValueError("no Output Name (*.hepmc) in run card")


def _override(raw, key, val):
    """Minimal in-place YAML scalar override for the two scan fields (Energy / KickMomentum)."""
    out = []
    for ln in raw.splitlines():
        s = ln.strip()
        if key == "Energy" and s.startswith("Energy:"):
            out.append(ln[: len(ln) - len(ln.lstrip())] + f"Energy: {val}")
        elif key == "KickMomentum" and s.startswith("KickMomentum:"):
            out.append(ln[: len(ln) - len(ln.lstrip())] + f"KickMomentum: [{val}, {val}]")
        else:
            out.append(ln)
    return "\n".join(out)


def seed_card(raw, seed):
    """Set the seed where each binary reads it, which is not the same place for the two of them.

    The neutrino generator reads Options/Initialize/Seed; the standalone cascade reads a top-level
    Initialize/seed.  A `Main: Seed` is read by neither, so setting only that leaves every shard
    identical and makes an N-shard production N copies of one sample.

    The cards pull Options from an included defaults file whose Seed is a fixed constant, so the
    include is expanded here with Seed replaced -- it cannot be overridden from the card otherwise.
    """
    m = re.search(r'^Options:\s*!include\s*"?([^"\n]+)"?\s*$', raw, re.M)
    if m:
        inc = m.group(1).strip()
        inc_host = achilles_data_root() / (inc.split("data/", 1)[-1] if inc.startswith("data/") else inc)
        opts = yaml.safe_load(Path(inc_host).read_text()) or {}
        opts.setdefault("Initialize", {})["Seed"] = int(seed)
        body = yaml.safe_dump(opts, default_flow_style=False, sort_keys=False)
        raw = raw[:m.start()] + "Options:\n" + "".join(f"  {ln}\n" for ln in body.splitlines()) + raw[m.end():]

    m2 = re.search(r'^Initialize:[ \t]*\n((?:[ \t]+\S.*\n)*)', raw, re.M)
    if not m2:
        return raw + f"\nInitialize:\n  seed: {seed}\n  Seed: {seed}\n"
    block = m2.group(0)
    new, n = re.subn(r'^([ \t]+)(seed|Seed):.*$', rf'\g<1>\g<2>: {seed}', block, flags=re.M)
    if not n:
        new = block.replace("Initialize:", f"Initialize:\n  seed: {seed}", 1)
    return raw[:m2.start()] + new + raw[m2.end():]


def run_one(card_path, out_dir=OUT_DIR, suffix="", overrides=None, dry=False):
    overrides = overrides or {}
    os.makedirs(out_dir, exist_ok=True)
    image, native, entry = _image_for(Path(card_path).name)
    card_name, hepmc = _write_card(card_path, out_dir, suffix, overrides)
    cmd = _RUNTIME_CMD[container_runtime()](card_name, image, native, entry, str(Path(out_dir).resolve()))
    print("  $", " ".join(cmd), flush=True)
    if dry:
        return None
    log = Path(out_dir) / f"{Path(card_name).stem}.log"
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    hp = Path(out_dir) / hepmc
    if not hp.exists():
        print(f"  WARNING: {hp} not produced (rc={rc}; see {log})", flush=True)
    return hp if hp.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run an ACHILLES run card in its container -> hepmc.")
    ap.add_argument("card")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--nevents", type=int, default=None, help="override NEvents (default: the card's)")
    ap.add_argument("--seeds", type=int, default=1, help="batches with distinct seeds")
    ap.add_argument("--seed0", type=int, default=1000, help="first seed")
    ap.add_argument("--scan-energy", help="comma list of beam energies [MeV] -> free-nucleon sigma(E)")
    ap.add_argument("--scan-kick", help="comma list of pi |p| [MeV] -> cascade sigma(p_pi)")
    ap.add_argument("--extract", help="run `python -m analysis.oracle_tools.extract <channel>` on each hepmc")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    a = ap.parse_args(argv)
    scan = [("Energy", v) for v in (a.scan_energy.split(",") if a.scan_energy else [])] + \
           [("KickMomentum", v) for v in (a.scan_kick.split(",") if a.scan_kick else [])]
    points = scan or [(None, None)]
    produced = []
    for key, val in points:
        for sd in range(a.seeds):
            sfx = (f"_{key.lower()}{val}" if key else "") + (f"_s{sd}" if a.seeds > 1 else "")
            ov = ({key: val} if key else {}) | {"seed": a.seed0 + sd, "nevents": a.nevents}
            hp = run_one(a.card, a.out_dir, sfx, ov, dry=a.dry_run)
            if hp:
                produced.append(hp)
                if a.extract:
                    out = str(hp).replace(".hepmc", ".npz")
                    subprocess.run([sys.executable, "-m", "analysis.oracle_tools.extract", a.extract, str(hp), out],
                                   check=False)
    print(f"DONE: {len(produced)} hepmc -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
