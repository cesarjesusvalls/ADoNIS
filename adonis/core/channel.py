"""Channel (primary interaction process) interface -- the sample/reweight contract.

Every channel separates:
  * `sample(key, n)`        -> detached Sample (the fixed proposal), and
  * `weight(params, S)`     -> (w, aux), pure-JAX and differentiable in `params`, and
  * `event_record(params, S)` -> EventRecord (full lab final state + weight + identities).

This is the kind-1 reweighting estimator made explicit: the proposal is fixed, only the
weight carries the knobs, so gradients are exact and a single sample can be reweighted /
differentiated over many parameter values.  New channels (2pi, QE, ...) subclass this.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from adonis.core.sample import check_sample
from adonis.core.validation import SelfTestMixin


class Channel(SelfTestMixin, ABC):
    sample_fields: tuple | None = None

    def validate_sample(self, sample):
        if self.sample_fields is not None:
            check_sample(sample, self.sample_fields, where=type(self).__name__)
        return sample

    @abstractmethod
    def sample(self, key, n):
        """Detached Sample bundle of `n` events."""

    @abstractmethod
    def weight(self, params, sample):
        """(weight (N,), aux) -- pure JAX, differentiable in `params`."""

    @abstractmethod
    def event_record(self, params, sample):
        """Full EventRecord (lab final state + weight + channel/PDG)."""

