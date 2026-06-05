"""No-FSI passthrough (bare production)."""
from __future__ import annotations

from adonis.fsi.base import FSIModel


class NoFSI(FSIModel):
    def apply(self, params, event):
        return event
