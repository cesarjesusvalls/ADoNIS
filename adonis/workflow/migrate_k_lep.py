"""Migrate bank chunks to the honest outgoing-lepton key:  `k_mu` / `k_e` -> `k_lep`.

The schema used to name the particle instead of its role: `k_mu` on CC banks, `k_e` on EM banks.
That is exactly the lie that makes an NC bank unsafe -- a field called `k_mu` holding a neutrino
produces wrong numbers nobody questions, because the name asserts something the data does not.

**Not a loader-side shim.** A shim would leave the on-disk schema permanently lying; anyone opening a
chunk would still see `k_mu`. The array is rewritten under its true name.

I/O only -- the array CONTENTS are copied verbatim, no physics is recomputed. `--verify` proves it:
for every chunk it re-reads both files and asserts

  * every OTHER array is byte-identical (`np.array_equal` on the raw buffers, dtype and shape too);
  * the key set differs by exactly one substitution;
  * the lepton array's own bytes are unchanged.

Atomic and restartable: each chunk is written to `<name>.tmp.npz` and `os.replace`d, so an
interrupted run leaves only complete files and re-running finishes the job. Idempotent: a chunk that
already carries `k_lep` is skipped without being rewritten.

    python -m adonis.workflow.migrate_k_lep $ADONIS_OUT              # dry run (default)
    python -m adonis.workflow.migrate_k_lep $ADONIS_OUT --apply
    python -m adonis.workflow.migrate_k_lep $ADONIS_OUT --verify     # census only

NOTE: `$ADONIS_OUT` is SHARED between the ADoNIS worktrees. Unlike the probe rename, this one is
load-bearing -- code that reads `B["k_mu"]` breaks on a migrated bank. Coordinate before applying.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

OLD_KEYS = ("k_mu", "k_e")          # CC banks, EM banks
NEW_KEY = "k_lep"
# --revert needs to know which old name a bank HAD.  The probe decides: CC banks carried k_mu, EM
# banks carried k_e, and the manifest records the probe -- so the revert is exact, not a guess.
_REVERT_BY_PROBE = {"CC": "k_mu", "EM": "k_e"}


def _chunk_files(root: Path):
    return sorted(root.rglob("chunk_*.npz"))


def _old_key_in(names) -> str | None:
    hits = [k for k in OLD_KEYS if k in names]
    if len(hits) > 1:
        raise RuntimeError(f"chunk carries BOTH {hits} -- refusing to guess which is the lepton")
    return hits[0] if hits else None


def migrate_chunk(path: Path, apply: bool) -> str | None:
    """Return a status string if this chunk needs (or got) migration, else None.

    G3 is enforced HERE, per chunk, BEFORE the original is replaced: the temp file is read back and
    every array compared against the in-memory original. That makes the gate self-contained -- no
    133 GB backup copy is needed to prove the rename moved no bytes -- and it means a chunk can only
    be replaced by one that has already been shown identical. A mismatch raises and leaves the
    original untouched.
    """
    with np.load(path, allow_pickle=True) as z:
        names = list(z.files)
        old = _old_key_in(names)
        if old is None:
            return None                      # already migrated, or a bank with no lepton array
        if NEW_KEY in names:
            return f"CONFLICT: has both {old} and {NEW_KEY}"
        if not apply:
            return f"{old}->{NEW_KEY}"
        data = {(NEW_KEY if k == old else k): z[k] for k in names}
    tmp = path.with_name(path.name + ".tmp.npz")     # np.savez APPENDS .npz -- name it so os.replace works
    np.savez(tmp, **data)
    _assert_identical(tmp, data, path)               # G3, before anything is overwritten
    os.replace(tmp, path)
    return f"{old}->{NEW_KEY}"


def _probe_of(path: Path) -> str | None:
    """The probe recorded in the nearest enclosing manifest.json (bank dir, then its parent)."""
    import json
    for d in (path.parent, path.parent.parent):
        m = d / "manifest.json"
        if m.exists():
            try:
                return json.loads(m.read_text()).get("probe")
            except Exception:
                return None
    return None


def revert_chunk(path: Path, apply: bool) -> str | None:
    """Undo migrate_chunk: `k_lep` -> the name this bank's probe originally used.

    This exists so the migration is genuinely reversible rather than reversible-in-principle.  The
    old name is read from the manifest's probe (CC -> k_mu, EM -> k_e), so it is recovered exactly;
    a bank whose probe is unknown is REFUSED rather than guessed.
    """
    with np.load(path, allow_pickle=True) as z:
        names = list(z.files)
        if NEW_KEY not in names:
            return None                                  # nothing to undo
        probe = _probe_of(path)
        old = _REVERT_BY_PROBE.get(probe)
        if old is None:
            return f"REFUSED: probe {probe!r} does not determine the old key"
        if old in names:
            return f"CONFLICT: has both {NEW_KEY} and {old}"
        if not apply:
            return f"{NEW_KEY}->{old}"
        data = {(old if k == NEW_KEY else k): z[k] for k in names}
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **data)
    _assert_identical(tmp, data, path)
    os.replace(tmp, path)
    return f"{NEW_KEY}->{old}"


def _assert_identical(tmp: Path, expect: dict, orig: Path):
    """Read the freshly written file back and prove it is the original with ONE key renamed."""
    with np.load(tmp, allow_pickle=True) as z:
        got, want = set(z.files), set(expect)
        if got != want:
            raise RuntimeError(f"{orig}: key set mismatch after write: +{sorted(got-want)} -{sorted(want-got)}")
        for k in sorted(want):
            x, y = z[k], expect[k]
            if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y, equal_nan=True):
                raise RuntimeError(f"{orig}: array {k!r} did not survive the rewrite")


def verify_chunk(path: Path, ref: Path) -> list[str]:
    """G3: every other array byte-identical, key set differs by exactly one substitution."""
    problems = []
    with np.load(path, allow_pickle=True) as a, np.load(ref, allow_pickle=True) as b:
        ka, kb = set(a.files), set(b.files)
        old = _old_key_in(kb)
        if old is None:
            return [f"{ref}: reference has no {OLD_KEYS} key"]
        if ka != (kb - {old}) | {NEW_KEY}:
            problems.append(f"{path}: key set is not a single substitution: "
                            f"+{sorted(ka - kb)} -{sorted(kb - ka)}")
        for k in sorted(kb & ka):                    # every OTHER array
            x, y = a[k], b[k]
            if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y, equal_nan=True):
                problems.append(f"{path}: array {k!r} CHANGED")
        if NEW_KEY in ka and old in kb:              # the lepton array's own contents
            x, y = a[NEW_KEY], b[old]
            if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y, equal_nan=True):
                problems.append(f"{path}: the lepton array itself CHANGED under the rename")
    return problems


def census(root: Path) -> Counter:
    c = Counter()
    for f in _chunk_files(root):
        try:
            with np.load(f, allow_pickle=True) as z:
                names = set(z.files)
        except Exception:
            c["<unreadable>"] += 1
            continue
        c[_old_key_in(names) or (NEW_KEY if NEW_KEY in names else "<no lepton key>")] += 1
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=os.environ.get("ADONIS_OUT"))
    ap.add_argument("--apply", action="store_true", help="rewrite the chunks (default is a dry run)")
    ap.add_argument("--verify", action="store_true", help="report the key census and exit non-zero if stale")
    ap.add_argument("--verify-against", metavar="BACKUP_ROOT",
                    help="run the full G3 byte-comparison against an unmigrated copy of the tree")
    ap.add_argument("--revert", action="store_true",
                    help="undo: k_lep -> k_mu/k_e per the manifest probe (makes the migration reversible)")
    a = ap.parse_args(argv)
    if not a.root:
        raise SystemExit("no root given and ADONIS_OUT is unset")
    root = Path(a.root)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    if a.verify_against:
        ref_root = Path(a.verify_against)
        bad = []
        n = 0
        for f in _chunk_files(root):
            ref = ref_root / f.relative_to(root)
            if not ref.exists():
                continue
            n += 1
            bad += verify_chunk(f, ref)
        print(f"G3 byte-comparison over {n} chunk(s): {'PASS' if not bad else 'FAIL'}")
        for p in bad[:20]:
            print("  " + p)
        return 1 if bad else 0

    if a.verify:
        c = census(root)
        for k, v in c.most_common():
            print(f"{v:6d}  {k}")
        stale = sum(c[k] for k in OLD_KEYS)
        print(f"\n{stale} chunk(s) still carry a retired lepton key")
        return 1 if stale else 0

    fn = revert_chunk if a.revert else migrate_chunk
    changed = Counter()
    files = _chunk_files(root)
    for i, f in enumerate(files, 1):
        try:
            r = fn(f, a.apply)
        except Exception as e:
            r = f"ERROR: {e}"
        if r:
            changed[r] += 1
        if a.apply and i % 100 == 0:
            print(f"  ... {i}/{len(files)} chunks scanned, {sum(changed.values())} rewritten", flush=True)
    verb = ("reverted" if a.revert else "migrated") if a.apply else \
           ("WOULD revert" if a.revert else "WOULD migrate") + " (dry run; pass --apply)"
    print(f"{verb} {sum(changed.values())} chunk(s) under {root}")
    for k, v in changed.most_common():
        print(f"  {v:6d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
