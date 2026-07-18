"""Generalized ACHILLES runner: configs/achilles/<card>.yml -> docker -> output/achilles/<...>.hepmc.

Mirrors the T2K generation for every validation observable.  Encodes the image/platform/entrypoint
rules from docs/CONTAINER.md (do NOT mix these up -- they are correctness, not preference):

  no-cascade runs  (inclusive (e,e'), free-nucleon, res1pi, T2K *_nofsi / T2K_H)
      -> ghcr.io/cesarjesusvalls/achilles:oracle , --platform linux/amd64 (emulation OK), bin/achilles
  FSI neutrino runs (T2K C/Ar fsi_gauss: Cascade Run:True)
      -> achilles:fullcascade (NATIVE arm64, NO --platform -- amd64+Rosetta SIGSEGVs the cascade), bin/achilles
  cascade-only pion runs (cascade_pip_*)
      -> achilles:cascade (native arm64, NO --platform), bin/cascade

Scans: free-nucleon sigma(E) overrides Beams[0]['Beam']['Beam Params']['Energy']; cascade sigma(p_pi)
overrides top-level 'KickMomentum'.  ACHILLES can SIGSEGV mid-run deterministically -> vary the seed per
batch and keep whatever each batch produced (partial hepmc is valid).

Usage (run on the host with docker; this module shells out to `docker run`):
  python -m analysis.utils.run_achilles configs/achilles/run_inclusive_ee_C_qe.yml
  python -m analysis.utils.run_achilles configs/achilles/run_freenucleon_res_H.yml --scan-energy 500,700,1000,1500,2000
  python -m analysis.utils.run_achilles configs/achilles/run_cascade_pip_C.yml --scan-kick 100,150,200,300,500
  python -m analysis.utils.run_achilles configs/achilles/run_T2K_C_fsi_gauss.yml --seeds 4 --extract cc1pi_rich
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


OUT_DIR = "output/achilles"
ORACLE = "ghcr.io/cesarjesusvalls/achilles:oracle"

# card-name prefix -> (image, native_arm64, entrypoint).  native_arm64=True => NO --platform.
_RULES = [
    # ANY cascade-mode card (Mode: CrossSection / Transparency), whatever the beam PID: pi+ (211),
    # proton (2212), neutron (2112).  The beam is a card setting (RunCascade.cc: m_pid = PID), so the
    # image/entrypoint must NOT be keyed on the beam species -- a run_cascade_prot card previously fell
    # through to the amd64 :oracle default, which SIGSEGVs the cascade under Rosetta.
    ("run_cascade", ("achilles:cascade", True, "/achilles/bin/achilles-cascade")),
    ("run_T2K_C_fsi",   ("achilles:fullcascade", True, "/achilles/bin/achilles")),  # Cascade Run:True
    ("run_T2K_Ar_fsi",  ("achilles:fullcascade", True, "/achilles/bin/achilles")),
    ("run_MINERvA_C_fsi",    ("achilles:fullcascade", True, "/achilles/bin/achilles")),  # different flux, C
    ("run_MicroBooNE_Ar_fsi", ("achilles:fullcascade", True, "/achilles/bin/achilles")),  # different flux+target
    # everything else is a no-cascade oracle run (inclusive, free-nucleon, res1pi, T2K *_nofsi, T2K_H)
]
_DEFAULT = (ORACLE, False, "/achilles/bin/achilles")


def _image_for(card_name):
    for prefix, rule in _RULES:
        if card_name.startswith(prefix):
            return rule
    return _DEFAULT


def _docker_cmd(card_in_out, image, native, entrypoint, out_abs):
    cmd = ["docker", "run", "--rm"]
    if not native:
        cmd += ["--platform", "linux/amd64"]            # amd64 :oracle under emulation (no-cascade only)
    cmd += ["-v", f"{out_abs}:/out", "--entrypoint", entrypoint, image, f"/out/{card_in_out}"]
    return cmd


def _write_card(card_path, out_dir, suffix, overrides):
    """Copy the run card into out_dir (the /out mount) with optional scan overrides.  The output hepmc keeps
    the card's ORIGINAL Output Name (so consumers find e.g. inclusive_ee_C_qe.hepmc), with `suffix` appended
    only for scan points / seeds so they don't clobber each other."""
    raw = Path(card_path).read_text()
    base_hepmc = _output_name(raw)                                 # e.g. /out/inclusive_ee_C_qe.hepmc
    hepmc_stem = Path(base_hepmc).name[: -len(".hepmc")]           # inclusive_ee_C_qe
    new_hepmc = f"/out/{hepmc_stem}{suffix}.hepmc"
    raw = raw.replace(base_hepmc, new_hepmc)
    for key, val in overrides.items():
        raw = _override(raw, key, val)
    card_name = Path(card_path).stem + suffix + ".yml"            # templated card file (keeps run_ prefix)
    (Path(out_dir) / card_name).write_text(raw)
    return card_name, new_hepmc.split("/")[-1]


def _output_name(raw):
    for ln in raw.splitlines():
        s = ln.strip()
        if s.startswith("Name:") and ".hepmc" in s:
            return s.split("Name:", 1)[1].strip()
    raise ValueError("no Output Name (*.hepmc) in run card")


def _override(raw, key, val):
    """Minimal in-place YAML scalar override for the two scan fields (Energy / KickMomentum) + Seed."""
    out = []
    for ln in raw.splitlines():
        s = ln.strip()
        if key == "Energy" and s.startswith("Energy:"):
            out.append(ln[: len(ln) - len(ln.lstrip())] + f"Energy: {val}")
        elif key == "KickMomentum" and s.startswith("KickMomentum:"):
            out.append(ln[: len(ln) - len(ln.lstrip())] + f"KickMomentum: [{val}, {val}]")
        elif key == "seed" and (s.startswith("seed:") or s.startswith("Seed:")):
            out.append(ln[: len(ln) - len(ln.lstrip())] + f"{s.split(':')[0]}: {val}")
        else:
            out.append(ln)
    return "\n".join(out)


def run_one(card_path, out_dir=OUT_DIR, suffix="", overrides=None, dry=False):
    overrides = overrides or {}
    os.makedirs(out_dir, exist_ok=True)
    image, native, entry = _image_for(Path(card_path).name)
    card_name, hepmc = _write_card(card_path, out_dir, suffix, overrides)
    cmd = _docker_cmd(card_name, image, native, entry, str(Path(out_dir).resolve()))
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
    ap = argparse.ArgumentParser(description="Run an ACHILLES run card in docker -> output/achilles/ hepmc.")
    ap.add_argument("card")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seeds", type=int, default=1, help="batches with distinct seeds (SIGSEGV resilience)")
    ap.add_argument("--scan-energy", help="comma list of beam energies [MeV] -> free-nucleon sigma(E)")
    ap.add_argument("--scan-kick", help="comma list of pi |p| [MeV] -> cascade sigma(p_pi)")
    ap.add_argument("--extract", help="run `python -m analysis.utils.extract <channel>` on each hepmc")
    ap.add_argument("--dry-run", action="store_true", help="print docker commands, do not run")
    a = ap.parse_args(argv)
    scan = [("Energy", v) for v in (a.scan_energy.split(",") if a.scan_energy else [])] + \
           [("KickMomentum", v) for v in (a.scan_kick.split(",") if a.scan_kick else [])]
    points = scan or [(None, None)]
    produced = []
    for key, val in points:
        for sd in range(a.seeds):
            sfx = (f"_{key.lower()}{val}" if key else "") + (f"_s{sd}" if a.seeds > 1 else "")
            ov = ({key: val} if key else {}) | {"seed": 1000 + sd}
            hp = run_one(a.card, a.out_dir, sfx, ov, dry=a.dry_run)
            if hp:
                produced.append(hp)
                if a.extract:
                    out = str(hp).replace(".hepmc", ".npz")
                    subprocess.run([sys.executable, "-m", "analysis.utils.extract", a.extract, str(hp), out],
                                   check=False)
    print(f"DONE: {len(produced)} hepmc -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
