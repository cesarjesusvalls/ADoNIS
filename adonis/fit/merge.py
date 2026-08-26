"""Validating a set of shards before merging them.

A merged product is only meaningful if its shards came from ONE run definition and cover ALL of the
work.  A NaN-fraction threshold on the merged grid is not a substitute for either check: it can accept
an incomplete scan silently backfilled at the wrong resolution, or merge shards whose axes agree but
whose injected truth, sigma or estimator do not.

This module checks the claims shards make about themselves (`adonis.fit.provenance`) and checks
coverage exactly, not statistically:

    rep = merge.check(files, arrays)
    rep.rows(covered=[(0, 7), (7, 7)], grid=21, what="M_A_res x Eb_shift")
    rep.raise_if_bad()          # or rep.emit() to warn only

A shard written before provenance stamping cannot be verified and is a WARNING rather than an error,
since it may not be re-runnable.  A stamped shard that disagrees with its siblings is always an ERROR:
once the information exists, ignoring it is how it rots.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from adonis.fit import provenance


# Every degradation knowingly accepted in this process, as (what, reason).  A figure is a PNG and has
# nowhere to keep a `prov_partial` field, so the fact lives here where the renderer can find it without
# every figure function having to pass a flag along.  See `analysis.paper.style.save`.
DEGRADED = []


@dataclass
class Report:
    what: str
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    digest: str = ""          # the agreed config digest, "" when unknown/unstamped
    git: str = ""
    _nw: int = 0              # how much of warnings/errors emit() has already printed
    _ne: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    # ---- checks ------------------------------------------------------------------------------------ #
    def rows(self, covered, grid, what=""):
        """Assert a set of (base, n) row blocks tiles range(grid) exactly.

        A statement about the WORK, not a proxy for it: which rows were assigned and which came back.
        A gap is named, and an overlap is reported too -- two shards writing the same rows means one of
        them computed something else than its filename claims.
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

    # ---- reporting --------------------------------------------------------------------------------- #
    def emit(self, printer=print):
        """Print what has not been printed yet.  A merge checks in stages -- provenance first, coverage
        once the candidates are grouped -- so this is called more than once and must not repeat itself."""
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
    """[0,1,2,7] -> '0-2,7'.  A gap of 300 rows should read as one range, not 300 numbers."""
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
