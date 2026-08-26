"""Migrate bank manifests to the honest probe names:  "weak" -> "CC",  "ee" -> "EM".

JSON ONLY.  Not one `.npz` is touched, so no physics number can move -- verified by the caller with
`--verify`, which re-reads every manifest and reports the resulting probe census.

Why this exists rather than a loader-side alias: an alias leaves the on-disk schema permanently
lying, and the next person to read a bank directory still sees "weak" on a file that may be NC.  The
rename is the fix; this script is how the existing data catches up with it.

Idempotent by construction: a manifest already carrying a current name is left byte-untouched, so
re-running is a no-op and a half-finished run can simply be re-run.

    python -m adonis.workflow.migrate_probe_names $ADONIS_OUT            # dry run (default)
    python -m adonis.workflow.migrate_probe_names $ADONIS_OUT --apply
    python -m adonis.workflow.migrate_probe_names $ADONIS_OUT --verify

NOTE for parallel worktrees: $ADONIS_OUT is SHARED between the ADoNIS checkouts.  Nothing reads
manifest["probe"] (it is write-only metadata -- grep before doubting it), so migrating is safe for a
checkout that has not renamed yet; but that checkout's generator will keep WRITING the old names
until it picks up this change.  Re-run --apply after those branches land.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

RENAME = {"weak": "CC", "ee": "EM"}


def migrate_file(path: Path, apply: bool) -> str | None:
    """Return "old->new" if this manifest needs (or got) a rename, else None."""
    try:
        raw = path.read_text()
        d = json.loads(raw)
    except Exception as e:
        return f"UNREADABLE ({e})"
    old = d.get("probe")
    if old not in RENAME:
        return None
    new = RENAME[old]
    if apply:
        d["probe"] = new
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(d, indent=2))
        os.replace(tmp, path)
    return f"{old}->{new}"


def census(root: Path) -> Counter:
    c = Counter()
    for m in root.rglob("manifest.json"):
        try:
            c[json.loads(m.read_text()).get("probe", "<none>")] += 1
        except Exception:
            c["<unreadable>"] += 1
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=os.environ.get("ADONIS_OUT"),
                    help="bank output root to walk (default: $ADONIS_OUT)")
    ap.add_argument("--apply", action="store_true", help="write the changes (default is a dry run)")
    ap.add_argument("--verify", action="store_true", help="report the probe census and exit non-zero if stale")
    a = ap.parse_args(argv)
    if not a.root:
        raise SystemExit("no root given and ADONIS_OUT is unset")
    root = Path(a.root)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    if a.verify:
        c = census(root)
        for k, v in c.most_common():
            print(f"{v:6d}  probe={k!r}")
        stale = sum(c[k] for k in RENAME)
        print(f"\n{stale} manifest(s) still carry a retired name")
        return 1 if stale else 0

    changed = Counter()
    for m in sorted(root.rglob("manifest.json")):
        r = migrate_file(m, a.apply)
        if r:
            changed[r] += 1
    verb = "migrated" if a.apply else "WOULD migrate (dry run; pass --apply)"
    total = sum(v for k, v in changed.items() if not k.startswith("UNREADABLE"))
    print(f"{verb} {total} manifest(s) under {root}")
    for k, v in changed.most_common():
        print(f"  {v:6d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
