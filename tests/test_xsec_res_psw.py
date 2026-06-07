"""RES phase-space weight (event.Weight) bit-exact vs the instrumented-ACHILLES RESDUMP.

Ports FinalStateMapper.cc::ThreeBodyMapper ((muN)+pi grouping: Masses() is lepton-first so the pion
is the t-channel-exchanged particle), HadronicMapper.cc::QESpectralMapper, and the process-dependent
BeamMapper seed.  Reconstructs psw = J_beam * J_had * (1/ThreeBodyGenerateWeight) to ~1e-12.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_res_psw import test_res_psw_bit_exact  # noqa: F401
