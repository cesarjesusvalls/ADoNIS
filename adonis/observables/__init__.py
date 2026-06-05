"""Observables: pure functions (EventRecord) -> (N,) for any channel / aggregate, plus the
OBSERVABLES registry.  (Currently in kinematics.py; leptonic/hadronic/tki split is cosmetic.)"""
from adonis.observables.kinematics import *          # noqa: F401,F403
from adonis.observables.kinematics import (           # noqa: F401
    OBSERVABLES, _mink_dot, _mag, _costheta)
