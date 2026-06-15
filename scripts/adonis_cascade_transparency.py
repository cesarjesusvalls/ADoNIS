"""ADoNIS pion-carbon transparency: sigma_reaction(p) and sigma_abs(p) from the PRODUCTION
differentiable cascade (`adonis.fsi.cascade_real.propagate`), matched to the ACHILLES
CrossSection-mode setup (pi+ on 12C, beam uniform in p over [80,500] MeV, impact parameter
uniform in a disk of radius R).

  sigma_reaction(p) = pi R^2 * P(interacted | p)      (scattered or absorbed)
  sigma_abs(p)      = pi R^2 * P(absorbed   | p)

Writes data/oracle/cascade_pip_c12_virt_abs_adonis.csv (same binning as the ACHILLES oracle).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax, jax.numpy as jnp

from adonis.fsi.cascade_real import RealCascadeConfig, propagate, _load_density
import adonis.fsi.oset_xsec as ox

NB = 14
LO, HI = 80.0, 500.0


def run(n_per_bin=1500, seed=0):
    rgrid, rho, _rhoN, radius = _load_density()
    b_max = float(radius)
    edges = np.linspace(LO, HI, NB + 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    key = jax.random.PRNGKey(seed)
    sig_r = np.zeros(NB); sig_a = np.zeros(NB)
    piR2 = np.pi * b_max ** 2 * 10.0          # fm^2 -> mb
    for b in range(NB):
        plo, phi = edges[b], edges[b + 1]
        n = n_per_bin
        key, ka, kb, kc, kp = jax.random.split(key, 5)
        # incident pi+ : p uniform in the bin, impact parameter uniform in the disk
        p_lab = jax.random.uniform(ka, (n,), minval=plo, maxval=phi)
        bimp = b_max * jnp.sqrt(jax.random.uniform(kb, (n,)))
        ang = 2 * jnp.pi * jax.random.uniform(kc, (n,))
        Epi = jnp.sqrt(ox.M_PIP ** 2 + p_lab ** 2)
        # start at the entry point on the nuclear surface (r = radius, just inside) heading +z,
        # so the cascade's r>radius escape does not fire before the pion enters
        z_entry = -jnp.sqrt(jnp.clip(radius ** 2 - bimp ** 2, 0.0, None)) * 0.999
        pos0 = jnp.stack([bimp * jnp.cos(ang), bimp * jnp.sin(ang), z_entry], axis=1)
        p_pi0 = jnp.stack([Epi, jnp.zeros(n), jnp.zeros(n), p_lab], axis=1)
        ch0 = jnp.zeros(n, jnp.int32)                          # 0 = pi+
        cfg = RealCascadeConfig(step=0.05, max_steps=int(2.5 * (b_max + 1) / 0.05))
        p_pi, ch, absorbed, nsc = propagate(pos0, p_pi0, ch0, cfg, kp, protfrac=0.0)
        absorbed = np.array(absorbed); nsc = np.array(nsc); ch = np.array(ch)
        reacted = absorbed | (nsc > 0) | (ch != 0)
        sig_a[b] = piR2 * absorbed.mean()
        sig_r[b] = piR2 * reacted.mean()
    return cen, sig_r, sig_a


if __name__ == "__main__":
    npb = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cen, sr, sa = run(npb)
    out = "data/oracle/cascade_pip_c12_virt_abs_adonis.csv"
    with open(out, "w") as f:
        f.write("# ADoNIS production differentiable cascade (cascade_real.propagate), pi+ on 12C\n")
        f.write("# matched to ACHILLES CrossSection setup; sigma = piR^2 * P(.|p), R from rho(r) tail\n")
        f.write("# p_pi[MeV]  sigma_reaction[mb]  sigma_abs[mb]\n")
        for c, r, a in zip(cen, sr, sa):
            f.write(f"{c:.1f}  {r:.2f}  {a:.2f}\n")
    print("wrote", out)
    print(f"# {'p':>6} {'sig_reac':>9} {'sig_abs':>8} {'absfrac':>8}")
    for c, r, a in zip(cen, sr, sa):
        print(f"{c:8.0f} {r:9.1f} {a:8.1f} {a/max(r,1e-9):8.3f}")
    print("p-avg abs fraction:", round(sa.sum()/sr.sum(), 3))
