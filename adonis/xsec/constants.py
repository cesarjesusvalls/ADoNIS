"""Back-compat re-export: the canonical constants module is ``adonis.constants``.

Exact ACHILLES `Constants.hh` values (MeV units throughout), reproduced to machine
precision so the transliterated cross section is on ACHILLES's absolute nb scale.
"""
from adonis.constants import (  # noqa: F401
    mp, mn, mN, mN2, mpip, mpi0, mdelta,
    HBARC, HBARC2, TO_NB,
    alpha, GF, GF_GEV, sin2w, cos2w, MZ, MW, GAMZ, GAMW, Vud, Vus, ee, cw, sw,
)
