"""A run outside the repository must find its ACHILLES references.

Configs name references as `output/achilles/...`, which is a path relative to the repository only
when the output root happens to be the default.  Resolving it against the working directory instead
of the configured root makes every reference vanish the moment a clone is run with ADONIS_OUT set.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_oracle_loaders_resolve_their_argument():
    """Each loader takes a config-supplied name, so each has to resolve it itself."""
    tree = ast.parse((ROOT / "adonis" / "workflow" / "selection.py").read_text())
    loaders = {"oracle_signal", "oracle_signal_nc", "ele_oracle_signal"}
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in loaders:
            src = ast.dump(node)
            assert "output_path" in src, f"{node.name} does not resolve its oracle path"
            seen.add(node.name)
    assert seen == loaders, f"missing loaders: {loaders - seen}"


def test_output_path_ignores_the_working_directory():
    """`output/x` names a file under the output root, not under wherever python was started."""
    import os
    from adonis.io import output_path
    old = os.environ.get("ADONIS_OUT")
    os.environ["ADONIS_OUT"] = "/somewhere/else"
    try:
        assert str(output_path("output/achilles/a.npz")) == "/somewhere/else/achilles/a.npz"
        assert str(output_path("/abs/a.npz")) == "/abs/a.npz"
    finally:
        os.environ.pop("ADONIS_OUT") if old is None else os.environ.update(ADONIS_OUT=old)
