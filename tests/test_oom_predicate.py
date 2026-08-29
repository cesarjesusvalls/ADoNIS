"""is_oom recognises every spelling XLA raises an out-of-memory under.

The spellings are not interchangeable: uppercasing the message maps "out of memory" to
"OUT OF MEMORY", which does not contain "OUT_OF_MEMORY", so a predicate written that way accepts the
underscored form and rejects the spaced one.
"""
from __future__ import annotations

import pytest

from adonis.fit.device_plan import is_oom, memory_note

OOM = [
    "RESOURCE_EXHAUSTED: Out of memory while trying to allocate 8589934592 bytes.",
    "XlaRuntimeError: RESOURCE_EXHAUSTED",
    "Execution of replica 0 failed: OUT_OF_MEMORY",
    "failed to allocate: out of memory",
    "CUDA error: out-of-memory",
]
NOT_OOM = [
    "INVALID_ARGUMENT: shapes must be equal",
    "NotImplementedError: no rule for primitive",
    "ValueError: dial order changed",
    "",
]


@pytest.mark.parametrize("msg", OOM)
def test_recognised(msg):
    assert is_oom(RuntimeError(msg)), msg


@pytest.mark.parametrize("msg", NOT_OOM)
def test_not_confused(msg):
    assert not is_oom(RuntimeError(msg)), msg


def test_memory_note_is_a_line_or_nothing():
    """On CPU the runtime reports no occupancy; callers must handle that without a branch of their own."""
    note = memory_note()
    assert note is None or ("GB in use" in note and "\n" not in note)
