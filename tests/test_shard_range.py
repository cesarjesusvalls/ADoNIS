"""shard_range reproduces the slicing the stages use, including the negative-base sentinel.

The output filename branches on that sentinel, so a shard that mistakes "all" for "from 0" overwrites
the whole-run file instead of writing its own.
"""
from __future__ import annotations

import pytest

from analysis._cli import shard_range


def _as_stages_write_it(base, count, total):
    return range(total) if base < 0 else range(base, min(base + count, total))


@pytest.mark.parametrize("total", [1, 4, 17, 41])
@pytest.mark.parametrize("base", [-1, 0, 1, 3, 16, 40])
@pytest.mark.parametrize("count", [1, 2, 5])
def test_matches_the_expression_it_replaces(base, count, total):
    lo, hi = shard_range(base, count, total)
    assert list(range(lo, hi)) == list(_as_stages_write_it(base, count, total))


def test_negative_base_is_the_whole_range(): 
    assert shard_range(-1, 1, 17) == (0, 17)


def test_a_shard_past_the_end_is_empty():
    lo, hi = shard_range(40, 5, 17)
    assert lo >= hi


def test_shards_tile_without_overlap():
    total, count = 17, 5
    seen = []
    for base in range(0, total, count):
        lo, hi = shard_range(base, count, total)
        seen += list(range(lo, hi))
    assert seen == list(range(total))
