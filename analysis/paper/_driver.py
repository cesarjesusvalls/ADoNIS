"""One entry point for the paper's figure groups.

`figures` maps a figure name to a callable taking the parsed command line, or to (module, function)
or (module, function, args_of); the two tuple forms import `module` and call `function`, with the run
label alone unless `args_of` builds the arguments.  `flags` names boolean options this group accepts
and forwards to the child.  `skip` reports, per figure, a reason it cannot run, which is not a failure.

Each figure renders in its own subprocess, which isolates its jax import and matplotlib state.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "adonis").is_dir())


@dataclass
class Args:
    """The command line, minus the figure names."""
    label: str | None = None
    flags: set = field(default_factory=set)
    extra: list = field(default_factory=list)   # positional `Key=value` arguments

    def as_argv(self):
        return ([] if self.label is None else ["--label", self.label]) + sorted(self.flags) + self.extra


def _callable(spec):
    if callable(spec):
        return spec
    mod, fn, *rest = spec
    args_of = rest[0] if rest else (lambda a: (a.label,))
    return lambda a: getattr(importlib.import_module(mod), fn)(*args_of(a))


def run(module, figures, label=None, flags=(), skip=None, argv=None):
    """Render the selected figures; `module` is this driver's own dotted path, for the subprocess."""
    argv = sys.argv[1:] if argv is None else list(argv)
    one = "--one" in argv
    a = Args(label=label, flags={f for f in flags if f in argv})
    if "--label" in argv:
        i = argv.index("--label")
        a.label = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    positional = [x for x in argv if not x.startswith("--")]
    a.extra = [x for x in positional if "=" in x]

    names = [x for x in positional if "=" not in x]
    keys = sorted(figures) if not names else [k for k in sorted(figures) if any(n in k for n in names)]
    if not keys:
        raise SystemExit(f"[figures] no figure matches {names}; known: {sorted(figures)}")

    if one:
        from analysis.paper import style
        style.use()
        for k in keys:
            _callable(figures[k])(a)
        return

    ok = ran = 0
    for k in keys:
        why = skip(k, a) if skip else None
        if why:
            print(f"  [skip {k}] {why}", flush=True)
            continue
        ran += 1
        print(f"\n=== {k}" + (f"  (label={a.label})" if a.label else "") + " ===", flush=True)
        cmd = [sys.executable, "-m", module, "--one", k, *a.as_argv()]
        ok += subprocess.run(cmd, cwd=str(ROOT)).returncode == 0
    print(f"\n{ok}/{ran} figures built", flush=True)
