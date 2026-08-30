"""The cascade record layout is chosen by the caller, never inherited from another module.

adonis/fsi/cascade.py keeps two record layouts: flat, where rec_caps is one budget for the whole
batch, and per-event, where rec_caps is the budget for a single event.  A caller that assumes one
and is handed the other overflows its buffers, so every generation path states which it wants at
the call.  Setting the module-level FLAT_FSI_REC from outside would make the layout depend on what
ran earlier in the process.
"""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent


def _calls(tree, name):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and (getattr(n.func, "attr", None) == name or getattr(n.func, "id", None) == name)]


def test_generation_states_its_record_layout():
    tree = ast.parse((SRC / "adonis/workflow/generate_bank.py").read_text())
    calls = _calls(tree, "cascade_nucleus") + _calls(tree, "run_cascade_pool")
    assert calls, "no cascade call found in generate_bank"
    for call in calls:
        kw = {k.arg for k in call.keywords}
        assert "rec_caps" not in kw or "flat_rec" in kw, (
            f"line {call.lineno}: passes rec_caps without saying which layout it sized them for")


def test_nothing_assigns_the_module_default():
    for path in SRC.glob("adonis/**/*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            targets = (node.targets if isinstance(node, ast.Assign) else
                       [node.target] if isinstance(node, ast.AnnAssign) else [])
            for t in targets:
                assert getattr(t, "attr", None) != "FLAT_FSI_REC", (
                    f"{path.relative_to(SRC)}:{node.lineno} sets FLAT_FSI_REC on another module; "
                    f"pass flat_rec at the call instead")
