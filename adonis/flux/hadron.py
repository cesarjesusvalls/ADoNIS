"""Tagged hadron beam (pi+/p/n) -- the incoming-particle description + ACHILLES CrossSection-mode
beam sampling.  A mono-species projectile fired along +z with |p| uniform in [pmin,pmax] at impact
parameter b uniform in a disk of radius R_DISK, starting at z0 = -1.05*R_nuc (5% outside the surface).

This is the reusable BEAM DESCRIPTION; the tagged-beam cascade STUDY (running the cascade + binning
sigma(p) etc.) lives in the analysis layer.
"""
import numpy as np

# beam key -> (PID, cascade species, charge)   charge: pion index {0:pi+,1:pi0,2:pi-}; nucleon 1=p, 0=n
BEAMS = {"pip": (211, "PION", 0), "prot": (2212, "NUCLEON", 1), "neut": (2112, "NUCLEON", 0)}

R_DISK = 10.0                                  # beam impact-parameter disk [fm] (card Params/radius)
PIR2_MB = np.pi * R_DISK ** 2 * 10.0           # pi R^2 in mb (1 fm^2 = 10 mb) = 3141.6 mb


class HadronBeam:
    """A tagged hadron projectile.  `mass` is the ON-SHELL mass the cascade uses for this species
    (so E^2 - m^2 = |p|^2 exactly); pass it in from the cascade mass tables."""

    def __init__(self, beam, mass):
        self.beam = beam
        self.pid, self.species, self.charge = BEAMS[beam]
        self.mass = float(mass)

    def sample(self, k_mom, k_disk, n, pmin, pmax, z0):
        """Draw n beam particles.  k_mom/k_disk: jax PRNGKeys (split by the caller, preserved for
        bit-for-bit).  z0: start-z [fm] = -1.05*R_nuc.  Returns (mom (n,), p4 (n,4), pos0 (n,3))."""
        import jax
        import jax.numpy as jnp
        mom = jax.random.uniform(k_mom, (n,), minval=pmin, maxval=pmax)          # TAGGED beam momentum
        u = jax.random.uniform(k_disk, (n, 2))
        br = R_DISK * jnp.sqrt(u[:, 0]); th = 2 * jnp.pi * u[:, 1]               # uniform in the disk
        E = jnp.sqrt(mom ** 2 + self.mass ** 2)
        p4 = jnp.stack([E, jnp.zeros(n), jnp.zeros(n), mom], axis=1)             # along +z
        pos0 = jnp.stack([br * jnp.cos(th), br * jnp.sin(th), jnp.full((n,), z0)], axis=1)
        return mom, p4, pos0
