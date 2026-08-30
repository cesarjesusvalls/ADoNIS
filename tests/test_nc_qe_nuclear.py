"""Nuclear NC-QE generator (qe_nc.generate): sane NC-elastic kinematics + validated absolute sigma.

The flux-averaged nuclear NC-QE sigma on 12C agrees with the ACHILLES oracle
(configs/achilles/run_inclusive_nc_C_qe.yml, sigma = 1.564e-5 nb) to 0.3% -- that absolute-scale gate is
the `slow` test below (needs the microboone flux table + jax).  The fast test pins the generator's
structural invariants (massless outgoing neutrino, elastic on BOTH species, no pion).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from adonis.flux.spectrum import set_default_flux
set_default_flux("flux/microboone_numu.dat")

import numpy as np
import pytest

from adonis.channels import qe_nc

_ACHILLES_ORACLE_NB = 1.564324879440945e-05


def test_generate_produces_sane_nc_elastic_events():
    ev = qe_nc.generate(20_000, material="C", seed=0, return_events=True, use_achilles_nc_coupling=True)["events"]
    w = np.asarray(ev["w"])
    assert np.isfinite(w).all() and w.sum() > 0
    kl = np.asarray(ev["k_lep"], float)
    m2 = kl[:, 0] ** 2 - np.sum(kl[:, 1:] ** 2, axis=1)
    assert np.median(np.abs(m2)) < 1.0
    ipid, Npid = np.asarray(ev["ipid"]), np.asarray(ev["Npid"])
    assert (ipid == Npid).all()
    assert (ipid == 2212).any() and (ipid == 2112).any()
    assert (np.asarray(ev["ppid"]) == 0).all()
    assert np.allclose(np.asarray(ev["p_pi"]), 0.0)


def test_theta_acc_cut_on_invisible_neutrino_is_refused():
    with pytest.raises(ValueError):
        qe_nc.generate(100, material="C", theta_acc=(0.0, 20.0))


@pytest.mark.slow
def test_nuclear_nc_qe_absolute_sigma_matches_achilles():
    """-nuclear: flux-averaged nuclear NC-QE sigma on 12C must match the ACHILLES oracle within 2%."""
    sig = qe_nc.generate(400_000, material="C", seed=0, return_events=False, use_achilles_nc_coupling=True)["sigma"]
    ratio = sig / _ACHILLES_ORACLE_NB
    assert 0.97 < ratio < 1.03, f"ADoNIS/ACHILLES nuclear NC-QE sigma ratio = {ratio:.4f} (sig={sig:.4e} nb)"
