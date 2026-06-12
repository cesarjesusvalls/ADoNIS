"""The Sample bundle and Sampler interface.

A `Sample` is the fixed, DETACHED proposal of the kind-1 estimator: everything that does
NOT depend on the physics knobs (kinematics, precomputed angular kernels, lepton tensor,
cuts, phase-space prefactors, lab final-state momenta).  It stays a plain dict at runtime
(a JAX pytree; some entries are static objects like the HadronStructure), but every
Channel declares its exact key set via `Channel.sample_fields` and the Generator checks
it once per proposal — the schema is enforced at the module seam, not by the container.

A `Sampler` produces a Sample from a PRNG key.  Composable samplers (flux, nuclear,
leptonic, decay-angle) contribute their detached draws into one bundle.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

Sample = dict


def check_sample(sample: Sample, fields, where: str = "Sample") -> Sample:
    """Validate `sample` against the declared schema `fields` (EXACT key set).

    Key-level only (cheap, trace-safe — runs at python level once per proposal);
    returns the sample unchanged so it can wrap call sites."""
    missing = set(fields) - sample.keys()
    extra = sample.keys() - set(fields)
    if missing or extra:
        raise KeyError(f"{where}: Sample schema mismatch -- "
                       f"missing {sorted(missing)}, unexpected {sorted(extra)}")
    return sample


class Sampler(ABC):
    @abstractmethod
    def propose(self, key, n) -> Sample:
        """Return a detached Sample bundle of `n` events."""
