"""Guard: the model->nb constant SIGMA_UNIT_NB is DERIVED (first principles), not fitted.
It must equal the closed-form bridge x 3-body-phase-space expression and stay ~2.51e-14
(the value validated vs the independent generator sigma to 0.5%).  Any reversion to a fitted
magic number, or a change to _NORM/couplings that isn't propagated, trips this test."""
import numpy as np
from adonis.primary.dcc.sigma_enu import SIGMA_UNIT_NB
from adonis.xsec import constants as C
from adonis.xsec.dcc_current import _NORM


def test_sigma_unit_is_derived_closed_form():
    bridge0 = (C.ee / (C.sw * np.sqrt(2.0))) ** 2 * 0.25 * (1.0 / C.MW ** 4) / _NORM
    expect = C.HBARC2 * C.TO_NB * 0.5 * bridge0 / (32.0 * C.mN * (2.0 * np.pi) ** 4 * 6.0)
    assert abs(SIGMA_UNIT_NB / expect - 1.0) < 1e-9, "SIGMA_UNIT_NB is not the derived closed form"
    assert 2.4e-14 < SIGMA_UNIT_NB < 2.6e-14, f"SIGMA_UNIT_NB={SIGMA_UNIT_NB:.4e} out of derived range"


if __name__ == "__main__":
    test_sigma_unit_is_derived_closed_form()
    print(f"OK: SIGMA_UNIT_NB = {SIGMA_UNIT_NB:.5e} is the derived closed form (zero fit)")
