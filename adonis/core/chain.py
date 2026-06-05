"""Generator: wire a flux + nuclear model + primary Channel (+ FSI) into one differentiable
event source.

The chain is the sample/reweight contract end to end: `channel.sample` draws the fixed
detached proposal (flux + nuclear + leptonic + decay angles), `channel.event_record`
applies the knob-dependent weight and builds the lab final state, and the FSI model
transforms it.  `generate(..., return_steps=True)` exposes the intermediate states for
step-by-step inspection.  Generation is chunked to bound memory (the per-event angular
kernels are large).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from adonis.params import PhysicsParams
from adonis.fsi.none import NoFSI
from adonis.core.event import EventRecord


def _concat_events(evs):
    return EventRecord(*[jnp.concatenate(xs, axis=0) for xs in zip(*evs)])


class Generator:
    """Compose a primary Channel with an FSI model. The flux/nuclear models live inside the
    channel (swappable there)."""

    def __init__(self, channel, fsi=None, params: PhysicsParams = PhysicsParams()):
        self.channel = channel
        self.fsi = fsi or NoFSI()
        self.params = params

    def generate(self, key, n, params: PhysicsParams | None = None, chunk=None,
                 return_steps=False):
        """Return an EventRecord of `n` events (weight differentiable in `params`).

        chunk: if set, generate in chunks of this size and concatenate (bounded memory).
        return_steps: also return {'sample', 'primary', 'post_fsi'} for chain inspection
        (only for a single, unchunked call)."""
        p = self.params if params is None else params
        if chunk and chunk < n:
            evs = []
            keys = jax.random.split(key, (n + chunk - 1) // chunk)
            done = 0
            for k in keys:
                m = min(chunk, n - done); done += m
                S = self.channel.sample(k, m)
                ev = self.fsi.apply(p, self.channel.event_record(p, S))
                evs.append(ev)
            return _concat_events(evs)
        S = self.channel.sample(key, n)
        primary = self.channel.event_record(p, S)
        post = self.fsi.apply(p, primary)
        if return_steps:
            return post, {"sample": S, "primary": primary, "post_fsi": post}
        return post
