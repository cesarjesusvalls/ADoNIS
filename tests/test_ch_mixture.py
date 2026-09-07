"""A mixed target must not be reweighted by how many chunks each element's bank has.

Every chunk's w0 already carries the full cross section (the generator divides by the chunk's own
event count), so a stream that ADDS chunks scales a component by its chunk count.  For a single
material that is an overall factor and cancels against a sigma defined as a fraction of the central
value; for CH it changes the carbon-to-hydrogen ratio, which is physics.
"""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "analysis" / "campaign" / "sample.py"


def _fn(name):
    tree = ast.parse(SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {SRC}")


def test_stream_grad_averages_over_chunks():
    """Without the division the free-H component scales with its own chunk count."""
    src = ast.dump(_fn("_stream_grad"))
    assert "n_ch" in src, "_stream_grad does not normalise by the number of chunks it read"


def test_free_h_is_streamed_with_the_same_chunk_limit():
    """Carbon and hydrogen must see the same max_chunks, or the ratio moves again."""
    src = ast.dump(_fn("_gradient"))
    assert src.count("max_chunks") >= 2, "the free-H stream does not take the carbon chunk limit"
