"""Phase I: exclusive electron vs neutrino single-pion production at MATCHED kinematics.

Both oracles are ACHILLES RES_Spectral_Func on 12C at E=2.222 GeV with the same forward
lepton-angle acceptance (AngleTheta [14,17] deg): electron scattering (EM, vector current)
vs nu_e CC (vector + axial).  Since the kinematics are identical, the difference in the
exclusive pion observables is the **axial current** (and its V-A interference): the pion
CM angular distribution cos(theta*) develops a forward-backward asymmetry for the neutrino
that the parity-conserving electron lacks, and the W / |p_pi| distributions shift.

Parses the two hepmc files (electron: inclusive_ee_12C_res.hepmc; nu_e:
exclusive_nue_12C_res.hepmc), builds an EventRecord per event and uses the standard
observables (cos_theta_star, W), and overlays the normalised distributions with a ratio
(nu/e) panel.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
from adonis.data.oracle.parse_hepmc import parse_events
from adonis.core.event import EventRecord
from adonis import observables as obs
from adonis.primary.dcc.channel import sample_final_state, weight_from_sample, assemble_event
from adonis.primary.dcc.structure import HadronStructure, CC_CHANNELS, EM_CHANNELS
from adonis.params import PhysicsParams
from adonis.core.validation import chi2_ndf
ROOT = Path(__file__).resolve().parents[1]
PI_PIDS = {111, 211, -211}
NMODEL = int(os.environ.get("ADONIS_EVNU_N", 60_000))


def model_exclusive(channels, current):
    """ADoNIS model (cos_theta_star, W, weight) for a current at the matched kinematics."""
    hs = HadronStructure(channels=channels, n_theta=12, n_phi=12, spline=False)
    S = sample_final_state(jax.random.PRNGKey(5), NMODEL, hs=hs, current=current, e_nu=2222.0,
                           ep_lo=50.0, ep_hi=2200.0, theta_max_deg=17.0, theta_min_deg=14.0)
    w, LWc = weight_from_sample(PhysicsParams(), S, use_spline=False)
    ev = assemble_event(S, w, LWc)
    return np.asarray(obs.cos_theta_star(ev)), np.asarray(obs.W(ev)), np.asarray(w)


def exclusive(path):
    """(cos_theta_star, W, Q2, weight) arrays for the single-pion events in a hepmc.

    Collect the per-event 4-momenta into arrays first, then call the observables ONCE
    (vectorised) -- per-event JAX boosts are ~1000x slower."""
    K, KP, PS, PPI, PN, WT = [], [], [], [], [], []
    for ev in parse_events(Path(path)):
        parts = ev["parts"]
        k_in = [p4 for pid, st, p4 in parts if abs(pid) in (11, 12) and st == 4]
        k_out = [p4 for pid, st, p4 in parts if pid == 11 and st == 1]
        p_pi = [p4 for pid, st, p4 in parts if pid in PI_PIDS and st == 1]
        p_N = [p4 for pid, st, p4 in parts if pid in (2112, 2212) and st == 1]
        p_str = [p4 for pid, st, p4 in parts if pid in (2112, 2212) and st == 2]
        if not (k_in and k_out and p_pi and p_N and p_str):
            continue
        K.append(k_in[0]); KP.append(k_out[0]); PS.append(p_str[0])
        PPI.append(p_pi[0]); PN.append(p_N[0]); WT.append(ev["w"])
    n = len(WT)
    z = jnp.zeros((n,))
    e = EventRecord(k=jnp.asarray(K), kp=jnp.asarray(KP), p_struck=jnp.asarray(PS),
                    p_pi=jnp.asarray(PPI), p_N=jnp.asarray(PN), w=jnp.ones(n),
                    channel=z, pid_pi=z, pid_N=z, pid_Ni=z, W=z, Q2_adj=z)
    return (np.asarray(obs.cos_theta_star(e)), np.asarray(obs.W(e)),
            np.asarray(obs.Q2(e)), np.asarray(WT))


def norm_hist(x, w, bins, rng):
    h, edges = np.histogram(x, bins=bins, range=rng, weights=w)
    h2, _ = np.histogram(x, bins=bins, range=rng, weights=w ** 2)
    c = 0.5 * (edges[1:] + edges[:-1]); s = h.sum()
    return c, h / s, (np.sqrt(h2) / s if s else h)


# binned oracle distributions: parse the hepmc if present (and save the committed CSV
# summaries), else load the CSVs.  Columns: center, e_shape, e_err, nu_shape, nu_err.
COS_BINS, COS_RNG = 16, (-1.0, 1.0)
W_BINS, W_RNG = 20, (1080.0, 1700.0)
_CSV_COS = ROOT / "data" / "oracle" / "exclusive_evnu_c12_cos.csv"
_CSV_W = ROOT / "data" / "oracle" / "exclusive_evnu_c12_W.csv"
_HEP_E = ROOT / "_oracle_out" / "inclusive_ee_12C_res.hepmc"
_HEP_N = ROOT / "_oracle_out" / "exclusive_nue_12C_res.hepmc"

if _HEP_E.exists() and _HEP_N.exists():
    cte, We, _, wte = exclusive(_HEP_E)
    ctn, Wn, _, wtn = exclusive(_HEP_N)
    for csv, x_e, x_n, b, r, lab in [(_CSV_COS, cte, ctn, COS_BINS, COS_RNG, "cos_theta_star"),
                                     (_CSV_W, We, Wn, W_BINS, W_RNG, "W[MeV]")]:
        c, he, ee = norm_hist(x_e, wte, b, r)
        _, hn, en = norm_hist(x_n, wtn, b, r)
        csv.write_text(f"# ACHILLES RES exclusive e/nu on 12C, E=2.222 GeV, theta_lep [14,17] deg.\n"
                       f"# {lab}_center, e_shape, e_err, nu_shape, nu_err (unit area)\n"
                       + "".join(f"{a:.4f} {b1:.5f} {c1:.5f} {d:.5f} {e:.5f}\n"
                                 for a, b1, c1, d, e in zip(c, he, ee, hn, en)))
        print("saved", csv)
    oc = (np.loadtxt(_CSV_COS), np.loadtxt(_CSV_W))
else:
    oc = (np.loadtxt(_CSV_COS), np.loadtxt(_CSV_W))

mcte, mWe, mwe = model_exclusive(EM_CHANNELS, "EM")
mctn, mWn, mwn = model_exclusive(CC_CHANNELS, "CC")


def panel(ax, axr, oracle, x_m_e, w_m_e, x_m_n, w_m_n, bins, rng, xlabel):
    """oracle: array [center, e_shape, e_err, nu_shape, nu_err]; model from raw arrays."""
    c, he, ee, hn, en = oracle.T
    _, hme, _ = norm_hist(x_m_e, w_m_e, bins, rng)
    _, hmn, _ = norm_hist(x_m_n, w_m_n, bins, rng)
    ax.errorbar(c, he, yerr=ee, fmt="bo", ms=3, label="e ACHILLES")
    ax.plot(c, hme, "b-", lw=1.5, label="e ADoNIS")
    ax.errorbar(c, hn, yerr=en, fmt="rs", ms=3, label=r"ν$_e$ ACHILLES")
    ax.plot(c, hmn, "r-", lw=1.5, label=r"ν$_e$ ADoNIS")
    ax.set_ylabel("norm. / bin"); ax.set_xticklabels([])
    c2e, nde = chi2_ndf(hme, np.zeros_like(hme), he, ee, floor=0.05)
    c2n, ndn = chi2_ndf(hmn, np.zeros_like(hmn), hn, en, floor=0.05)
    ge = he > 0.05 * he.max(); gn = hn > 0.05 * hn.max()
    axr.plot(c[ge], hme[ge] / he[ge], "b^", ms=4)
    axr.plot(c[gn], hmn[gn] / hn[gn], "rv", ms=4)
    axr.axhline(1, ls="--", color="gray"); axr.set_ylabel("ADoNIS/ACH"); axr.set_xlabel(xlabel)
    axr.set_ylim(0.5, 1.5)
    return c2e / max(nde, 1), c2n / max(ndn, 1)


def fb_from_hist(o):
    c, he, _, hn, _ = o.T; pos = c > 0; neg = c < 0
    return ((he[pos].sum() - he[neg].sum()) / he.sum(),
            (hn[pos].sum() - hn[neg].sum()) / hn.sum())


def mean_from_hist(o):
    c, he, _, hn, _ = o.T
    return np.average(c, weights=he), np.average(c, weights=hn)


fig, axes = plt.subplots(2, 2, figsize=(12, 7), height_ratios=[3, 1])
x2e_c, x2n_c = panel(axes[0, 0], axes[1, 0], oc[0], mcte, mwe, mctn, mwn, 16, (-1, 1),
                     r"$\cos\theta^*_\pi$")
afe, afn = fb_from_hist(oc[0])
axes[0, 0].set_title(r"pion CM angle $\cos\theta^*_\pi$  (e A_FB=%+.3f, ν=%+.3f)" % (afe, afn))
axes[0, 0].legend(fontsize=8, ncol=2, title=f"χ²/ndf  e:{x2e_c:.1f}  ν:{x2n_c:.1f}")
x2e_w, x2n_w = panel(axes[0, 1], axes[1, 1], oc[1], mWe, mwe, mWn, mwn, 20, (1080, 1700),
                     "W [MeV]")
we_m, wn_m = mean_from_hist(oc[1])
axes[0, 1].set_title(r"hadronic invariant mass W  (⟨W⟩ e=%.0f, ν=%.0f)" % (we_m, wn_m))
axes[0, 1].legend(fontsize=8, ncol=2, title=f"χ²/ndf  e:{x2e_w:.1f}  ν:{x2n_w:.1f}")

fig.suptitle(r"Exclusive e vs ν$_e$ single-π on $^{12}$C, matched kinematics (E=2.222 GeV, "
             r"$\theta_\ell\in[14,17]°$) — ADoNIS vs ACHILLES RES")
fig.tight_layout()
out = ROOT / "figures"; out.mkdir(exist_ok=True)
fig.savefig(out / "exclusive_evnu_c12.png", dpi=130)
print("wrote", out / "exclusive_evnu_c12.png")
print(f"cos*  A_FB (oracle):  e {afe:+.3f}   nu {afn:+.3f}")
print(f"<W> (oracle):  e {we_m:.0f}   nu {wn_m:.0f} MeV")
print(f"chi2/ndf model-vs-oracle: cos* e {x2e_c:.1f} nu {x2n_c:.1f}; W e {x2e_w:.1f} nu {x2n_w:.1f}")
