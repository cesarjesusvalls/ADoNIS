"""Validating a set of shards before merging them.

Checks that shards came from ONE run definition and cover ALL of the intended work, via each
shard's stamped provenance (`adonis.fit.provenance`) and an exact coverage check over row indices:

    rep = merge.check(files, arrays)
    rep.rows(covered=[(0, 7), (7, 7)], grid=21, what="<axis description>")
    rep.raise_if_bad()          # or rep.emit() to warn only

A shard with no provenance stamp is a WARNING (it predates stamping and may not be re-runnable); a
stamped shard that disagrees with its siblings is always an ERROR.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from adonis.fit import provenance


DEGRADED = []


@dataclass
class Report:
    what: str
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    digest: str = ""
    git: str = ""
    _nw: int = 0
    _ne: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def rows(self, covered, grid, what=""):
        """Assert a set of (base, n) row blocks tiles range(grid) exactly: every row covered once,
        none outside the grid.  A gap is reported as missing rows; an overlap (two shards writing the
        same row) is reported as a warning.
        """
        seen = {}
        for base, n in covered:
            for r in range(int(base), int(base) + int(n)):
                seen[r] = seen.get(r, 0) + 1
        missing = [r for r in range(int(grid)) if r not in seen]
        dup = sorted(r for r, c in seen.items() if c > 1)
        over = sorted(r for r in seen if r >= int(grid))
        tag = f"{what}: " if what else ""
        if missing:
            self.error(f"{tag}rows not computed: {_runs(missing)} of {grid}")
        if dup:
            self.warn(f"{tag}rows written by more than one shard: {_runs(dup)}")
        if over:
            self.error(f"{tag}rows outside the grid: {_runs(over)} (grid is {grid})")
        return not missing and not over

    def incomplete(self, names):
        """Shards whose own `complete` flag is False -- checkpointed but preempted before finishing."""
        if names:
            self.error(f"{len(names)} shard(s) did not finish (complete=False): {sorted(names)[:6]}"
                       + (" ..." if len(names) > 6 else ""))

    def emit(self, printer=print):
        """Print what has not been printed yet.  Safe to call more than once (e.g. once per check
        stage) without repeating earlier output."""
        for w in self.warnings[self._nw:]:
            printer(f"[warn] {self.what}: {w}")
        for e in self.errors[self._ne:]:
            printer(f"[ERROR] {self.what}: {e}")
        self._nw, self._ne = len(self.warnings), len(self.errors)
        return self

    def raise_if_bad(self, allow_partial=False):
        self.emit()
        if self.errors and not allow_partial:
            raise SystemExit(
                f"refusing to merge {self.what}: " + "; ".join(self.errors)
                + "\n\nThe merged product would be built from an incomplete or inconsistent shard set. "
                  "Finish or delete the offending shards, or pass --allow-partial to accept a product "
                  "that is knowingly partial.")
        if self.errors:
            print(f"[warn] {self.what}: --allow-partial given; merging anyway")
            DEGRADED.append((self.what, "; ".join(self.errors)))
        return self

    def stamp(self) -> dict:
        """Provenance to carry into the MERGED npz, so a figure can say what it is plotting."""
        return {"prov_digest": self.digest, "prov_git": self.git,
                "prov_partial": bool(self.errors), "prov_merged_from": self.what}


def check(files, arrays, what="") -> Report:
    """Cross-shard consistency: one config digest, one commit, and a note on what is unverifiable.

    `files` are paths (for messages), `arrays` the loaded npz objects, in the same order.
    """
    rep = Report(what=what or f"{len(files)} shard(s)")
    seen, unstamped, incomplete = {}, [], []
    for f, z in zip(files, arrays):
        p = provenance.read(z)
        if p is None:
            unstamped.append(_short(f))
        else:
            seen.setdefault((p.get("digest", ""), p.get("config", "")), []).append(_short(f))
            rep.git = rep.git or p.get("git", "")
        if "complete" in getattr(z, "files", ()) and not bool(z["complete"]):
            incomplete.append(_short(f))

    if unstamped:
        rep.warn(f"{len(unstamped)} shard(s) carry no provenance and cannot be verified "
                 f"(written before stamping): {sorted(unstamped)[:4]}"
                 + (" ..." if len(unstamped) > 4 else ""))
    if len(seen) > 1:
        detail = "; ".join(f"{d or '(none)'} <- {sorted(v)[:3]}" for (d, _c), v in sorted(seen.items()))
        rep.error(f"shards come from {len(seen)} different run definitions: {detail}")
    elif seen:
        rep.digest = next(iter(seen))[0]
    rep.incomplete(incomplete)
    return rep


def _runs(xs) -> str:
    """[0,1,2,7] -> '0-2,7': collapses consecutive integers into ranges for compact messages."""
    xs = sorted(xs)
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(f"{xs[i]}" if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ",".join(out)


def _short(f) -> str:
    import os
    return os.path.basename(str(f))
