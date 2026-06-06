"""Phase G gate: the ACHILLES cascade reaction cross section reproduces the Delta(1232).

Produced by the cascade-enabled image we built (docker/Dockerfile.cascade -> achilles-cascade),
Virtual-Resonances mode, pi+ on 12C: the reaction sigma(p_pi) shape (N_hits(p), since the
beam is uniform in p) must peak in the Delta region (p_pi ~ 240-320 MeV, T_pi ~ 150-200 MeV).
This is the model-INDEPENDENT cascade oracle for Fig c12_ar40. See docs/phases/PHASE_G.md.
"""
from pathlib import Path

import numpy as np
import pytest

_CSV = Path(__file__).resolve().parent.parent / "data" / "oracle" / "cascade_pip_c12_reaction.csv"
pytestmark = pytest.mark.skipif(not _CSV.exists(), reason=f"cascade oracle not present ({_CSV})")


def test_cascade_reaction_delta_peak():
    ref = np.loadtxt(_CSV)
    p, sig = ref[:, 0], ref[:, 1]
    ipk = int(np.argmax(sig))
    assert 240 <= p[ipk] <= 320, p[ipk]                 # Delta(1232) reaction peak
    # resonance shape: rises into the Delta and falls above it
    assert np.interp(110, p, sig) < sig[ipk]
    assert np.interp(480, p, sig) < sig[ipk]
