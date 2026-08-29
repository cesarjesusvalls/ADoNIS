"""Pollutes the environment at import, the way a module that generates a bank does."""
import os

os.environ["ADONIS_FLUX_FILE"] = "flux/minerva_numu_fhc.dat"


def test_placeholder():
    assert True
