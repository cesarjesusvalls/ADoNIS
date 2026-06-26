"""Phase G/H + D: the pion-propagation (cascade FSI) effect on the neutrino final state.

Compares ACHILLES nu_e RES single-pion production on 12C WITHOUT vs WITH the in-event cascade
(Cascade: Run: False vs True), at the same kinematics, into:
  * the final-state pion multiplicity -> the CC1pi -> CC0pi absorption fraction;
  * the leading-pion momentum spectrum |p_pi| -> the FSI energy-loss / re-scattering softening.

Also overlays the ADoNIS ToyCascadeFSI applied to the produced (cascade-off) pions, to show
the differentiable toy cascade reproduces the QUALITATIVE propagation effect (absorption +
softening).  The cascade-on oracle needs the main `achilles` binary with the in-event cascade
fix (docker/Dockerfile.fullcascade -> achilles:fullcascade); see PHASE_G.md.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from analysis.utils.hepmc import parse_events
from adonis.core.event import EventRecord
from adonis.fsi.cascade import ToyCascadeFSI, CascadeConfig
from adonis.core.params import PhysicsParams
ROOT = Path(__file__).resolve().parents[1]
PI_PIDS = {111, 211, -211}


def pion_obs(path):
    """Per event: (n_pi final-state pions, leading-pion |p| [MeV], weight)."""
    npi, plead, wt = [], [], []
    for ev in parse_events(Path(path)):
        pis = [p4 for pid, st, p4 in ev["parts"] if pid in PI_PIDS and st == 1]
        npi.append(len(pis))
        plead.append(max((np.linalg.norm(p[1:]) for p in pis), default=0.0))
        wt.append(ev["w"])
    return np.array(npi), np.array(plead), np.array(wt)


OFF = ROOT / "_oracle_out" / "exclusive_nue_12C_res.hepmc"
ON = ROOT / "_oracle_out" / "nue_12C_cascade.hepmc"
_CSV = ROOT / "data" / "oracle" / "cascade_effect_nue_c12.csv"
EDGES = np.linspace(0, 600, 31); CTR = 0.5 * (EDGES[1:] + EDGES[:-1])
noff, poff, woff = pion_obs(OFF)
non, pon, won = pion_obs(ON)

cc0_off = woff[noff == 0].sum() / woff.sum()
cc0_on = won[non == 0].sum() / won.sum()

# persist the binned off/on leading-pion spectra (committable, CI-runnable summary)
def _nh(p, w):
    h, _ = np.histogram(p, bins=EDGES, weights=w)
    h2, _ = np.histogram(p, bins=EDGES, weights=w ** 2)
    s = h.sum(); return h / s, np.sqrt(h2) / s
hoff, eoff = _nh(poff[noff == 1], woff[noff == 1])
honn, eonn = _nh(pon[non == 1], won[non == 1])
mean_off = np.average(poff[noff == 1], weights=woff[noff == 1])
mean_on = np.average(pon[non == 1], weights=won[non == 1])
_CSV.write_text(
    f"# ACHILLES nu_e RES single-pi on 12C, E=2.222 GeV: cascade OFF vs ON leading-pion |p|.\n"
    f"# cc0pi_off={cc0_off:.4f} cc0pi_on={cc0_on:.4f} mean_off={mean_off:.1f} mean_on={mean_on:.1f}\n"
    f"# p_center[MeV], off_shape, off_err, on_shape, on_err (unit area, CC1pi)\n"
    + "".join(f"{c:.1f} {a:.5f} {b:.5f} {d:.5f} {e:.5f}\n"
             for c, a, b, d, e in zip(CTR, hoff, eoff, honn, eonn)))
print("saved", _CSV)

# ADoNIS ToyCascadeFSI applied to the cascade-off produced pions (qualitative model overlay)
sel = noff == 1
n = int(sel.sum())
dirs = np.random.default_rng(0).normal(size=(n, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
M_PI = 139.57
p3 = jnp.asarray(dirs * poff[sel][:, None]); E = jnp.sqrt(poff[sel] ** 2 + M_PI ** 2)
z = jnp.zeros((n, 4))
ev0 = EventRecord(k=z, kp=z, p_struck=z, p_pi=jnp.concatenate([jnp.asarray(E)[:, None], p3], 1),
                  p_N=z, w=jnp.asarray(woff[sel]), channel=jnp.zeros(n, jnp.int32),
                  pid_pi=jnp.full(n, 211, jnp.int32), pid_N=jnp.full(n, 2212, jnp.int32),
                  pid_Ni=jnp.full(n, 2112, jnp.int32), W=jnp.full(n, 1232.0), Q2_adj=jnp.full(n, 1e5))
# sigma tuned to the ACHILLES cascade (CC0pi fraction + softening): the differentiable toy
# FSI matches the real cascade's gross effects via its two physical knobs.
fsi = ToyCascadeFSI(CascadeConfig(seed=3, oset_shape=True))
ev1 = fsi.apply(PhysicsParams(fsi_sigma_scatter=0.17, fsi_sigma_abs=0.10), ev0, key=jax.random.PRNGKey(1))
m_abs = np.asarray(ev1.pid_pi == 0)
m_pmag = np.asarray(jnp.linalg.norm(ev1.p_pi[:, 1:], axis=1))
m_w = np.asarray(ev1.w)
cc0_model = float(m_w[m_abs].sum() / woff[sel].sum())

edges = np.linspace(0, 600, 31); ctr = 0.5 * (edges[1:] + edges[:-1])
ho, _ = np.histogram(poff[noff == 1], bins=edges, weights=woff[noff == 1])
hn, _ = np.histogram(pon[non == 1], bins=edges, weights=won[non == 1])
hm, _ = np.histogram(m_pmag[~m_abs], bins=edges, weights=m_w[~m_abs])

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax.step(ctr, ho / ho.sum(), where="mid", lw=2, label="cascade OFF (produced)")
ax.step(ctr, hn / hn.sum(), where="mid", lw=2, label="cascade ON (ACHILLES)")
ax.step(ctr, hm / hm.sum(), where="mid", lw=1.5, ls="--", color="k", label="ADoNIS ToyCascadeFSI")
ax.set_xlabel(r"leading-$\pi$ $|p_\pi|$ [MeV]"); ax.set_ylabel("norm. CC1π / bin")
ax.set_title("pion spectrum: FSI softening"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

labels = ["CC0π\n(absorbed)", "CC1π", "CC≥2π"]
off_frac = [cc0_off, woff[noff == 1].sum() / woff.sum(), woff[noff >= 2].sum() / woff.sum()]
on_frac = [cc0_on, won[non == 1].sum() / won.sum(), won[non >= 2].sum() / won.sum()]
x = np.arange(3)
ax2.bar(x - 0.2, off_frac, 0.4, label="cascade OFF")
ax2.bar(x + 0.2, on_frac, 0.4, label="cascade ON")
ax2.set_xticks(x); ax2.set_xticklabels(labels); ax2.set_ylabel("fraction")
ax2.set_title(f"pion topology (ACHILLES CC0π {cc0_on:.0%}; ADoNIS {cc0_model:.0%})")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")
fig.suptitle(r"Pion propagation (cascade FSI) on ν$_e$ single-π, $^{12}$C — ACHILLES off vs on")
fig.tight_layout()
out = ROOT / "figures"; out.mkdir(exist_ok=True)
fig.savefig(out / "cascade_effect_nue_c12.png", dpi=130)
print("wrote", out / "cascade_effect_nue_c12.png")
print(f"CC0pi fraction: cascade OFF {cc0_off:.3f}  ON {cc0_on:.3f}  ADoNIS toy {cc0_model:.3f}")
print(f"<|p_pi|> CC1pi: OFF {np.average(poff[noff==1],weights=woff[noff==1]):.0f}  "
      f"ON {np.average(pon[non==1],weights=won[non==1]):.0f} MeV")
