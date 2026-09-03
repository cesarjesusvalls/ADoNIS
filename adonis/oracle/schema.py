"""The observable file ADoNIS compares against, and how to check one.

A comparison needs both generators to answer the same question, not to share an implementation.  The
answer is one .npz of per-event observables and weights; anything that writes this file can be
compared, whichever generator produced it.  `adonis.workflow.selection.oracle_signal` and its NC and
electron variants read exactly these fields.

This describes the combined per-sample file, the one a comparison reads.  Per-shard intermediates
carry the same per-event fields but no `weight_to_nb`: they are not comparable until combined.

Weights are per event in whatever unit the generator used; `weight_to_nb` converts their sum to a
cross section in nb.  Four-vectors are (E, px, py, pz) in MeV.  Per-event particle lists are padded
to a fixed width, with unused slots zero and, for identifiers, 0 meaning "no particle".

    python -m adonis.oracle.schema <file.npz> [...]
"""
from __future__ import annotations

import sys

import numpy as np

COMMON = {
    "w":            ("(n,)",     "per-event weight"),
    "weight_to_nb": ("scalar",   "multiplier taking sum(w) to a cross section in nb"),
}

CHARGED_LEPTON = {
    "lep":          ("(n, 4)",   "outgoing charged lepton four-vector"),
}

HADRONS = {
    "prot_p4":      ("(n, k, 4)", "outgoing proton four-vectors, zero-padded"),
    "pi_p4":        ("(n, m, 4)", "outgoing pion four-vectors, zero-padded"),
    "pi_pid":       ("(n, m)",    "PDG code per pion slot, 0 where empty"),
}

NC_MESONS = {
    "n_nonpion_meson":  ("(n,)", "mesons that are not pions; counting pi0 here would veto the signal"),
}

OPTIONAL = {
    "n_other_meson":    ("(n,)", "non-pion mesons per event; absent is treated as none"),
    "proc":             ("(n,)", "generator process code; absent means every event is signal"),
    "n_pi_out":         ("(n,)", "outgoing pion multiplicity"),
}

KINDS = {
    "cc":  {**COMMON, **CHARGED_LEPTON, **HADRONS},
    "nc":  {**COMMON, **HADRONS, **NC_MESONS},
    "ele": {**COMMON, **CHARGED_LEPTON},
}


def problems(path, kind="cc"):
    """Every way `path` fails the schema for `kind`, as a list of sentences.  Empty means it passes."""
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} not in {sorted(KINDS)}")
    d = np.load(path, allow_pickle=True)
    out = []
    required = KINDS[kind]
    for name, (shape, what) in required.items():
        if name not in d.files:
            out.append(f"missing {name} ({shape}): {what}")
    if out:
        return out

    n = len(np.atleast_1d(d["w"]))
    for name in required:
        if name == "weight_to_nb":
            continue
        a = np.asarray(d[name])
        if a.shape[0] != n:
            out.append(f"{name} has {a.shape[0]} rows, but w has {n}")
    for name, ax in (("lep", 2), ("prot_p4", 3), ("pi_p4", 3), ("pi_pid", 2)):
        if name in required and name in d.files and np.asarray(d[name]).ndim != ax:
            out.append(f"{name} is {np.asarray(d[name]).ndim}-dimensional, expected {ax}")
    if "lep" in required and "lep" in d.files and np.asarray(d["lep"]).shape[-1] != 4:
        out.append("lep four-vectors are not length 4")
    if float(np.asarray(d["weight_to_nb"])) <= 0:
        out.append("weight_to_nb is not positive, so the sum of weights has no cross-section meaning")
    if not np.all(np.isfinite(np.asarray(d["w"], float))):
        out.append("w contains non-finite entries")
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    kind = "cc"
    if "--kind" in argv:
        i = argv.index("--kind"); kind = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    if not argv:
        raise SystemExit(__doc__.strip().splitlines()[-1].strip())
    bad = 0
    for path in argv:
        issues = problems(path, kind)
        print(f"{path}: {'ok' if not issues else 'FAILS'}")
        for s in issues:
            print(f"    {s}")
        bad += bool(issues)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
