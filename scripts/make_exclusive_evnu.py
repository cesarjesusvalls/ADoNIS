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


def model_exclusive(channels, current, m_lep=0.0):
    """ADoNIS model (cos_theta_star, W, weight) for a current at the matched kinematics.
    `m_lep` is the outgoing charged-lepton mass (0 for e/nu_e, 105.658 MeV for nu_mu)."""
    hs = HadronStructure(channels=channels, n_theta=12, n_phi=12, spline=False)
    S = sample_final_state(jax.random.PRNGKey(5), NMODEL, hs=hs, current=current, e_nu=2222.0,
                           ep_lo=50.0, ep_hi=2200.0, theta_max_deg=17.0, theta_min_deg=14.0,
                           m_lep=m_lep)
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
        k_in = [p4 for pid, st, p4 in parts if abs(pid) in (11, 12, 14) and st == 4]
        k_out = [p4 for pid, st, p4 in parts if abs(pid) in (11, 13) and st == 1]
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
_HEP = {"e": ROOT / "_oracle_out" / "inclusive_ee_12C_res.hepmc",
        "nue": ROOT / "_oracle_out" / "exclusive_nue_12C_res.hepmc",
        "numu": ROOT / "_oracle_out" / "exclusive_numu_12C_res.hepmc"}

if all(p.exists() for p in _HEP.values()):
    # parse all three, write 7-col CSVs: center, {e,nue,numu} x {shape, err}
    raw = {k: exclusive(p) for k, p in _HEP.items()}        # (cos, W, Q2, wt)
    for csv, idx, b, r, lab in [(_CSV_COS, 0, COS_BINS, COS_RNG, "cos_theta_star"),
                                (_CSV_W, 1, W_BINS, W_RNG, "W[MeV]")]:
        cols = []
        for k in ("e", "nue", "numu"):
            c, h, e = norm_hist(raw[k][idx], raw[k][3], b, r)
            cols += [h, e]
        center = c
        csv.write_text(f"# ACHILLES RES exclusive e/nu_e/nu_mu on 12C, E=2.222 GeV, theta_lep [14,17] deg.\n"
                       f"# {lab}_center, e_shape, e_err, nue_shape, nue_err, numu_shape, numu_err (unit area)\n"
                       + "".join(" ".join(f"{v:.5f}" for v in row) + "\n"
                                 for row in zip(center, *cols)))
        print("saved", csv)
oc = (np.loadtxt(_CSV_COS), np.loadtxt(_CSV_W))

mcte, mWe, mwe = model_exclusive(EM_CHANNELS, "EM")
mctn, mWn, mwn = model_exclusive(CC_CHANNELS, "CC")
mctm, mWm, mwm = model_exclusive(CC_CHANNELS, "CC", m_lep=105.658)


SERIES = [("e", "b", "o", "EM", mcte, mWe, mwe), ("ν$_e$", "r", "s", "CC", mctn, mWn, mwn),
          ("ν$_μ$", "g", "^", "CC", mctm, mWm, mwm)]


def panel(ax, axr, oracle, which, bins, rng, xlabel):
    """oracle cols: center, {e,nue,numu}x{shape,err}.  which: 0 cos*, 1 W (selects model arr)."""
    c = oracle[:, 0]; chis = {}
    for j, (lab, col, mk, cur, mcos, mW, mw) in enumerate(SERIES):
        ho, eo = oracle[:, 1 + 2 * j], oracle[:, 2 + 2 * j]
        _, hm, _ = norm_hist(mcos if which == 0 else mW, mw, bins, rng)
        ax.errorbar(c, ho, yerr=eo, fmt=col + mk, ms=3, label=f"{lab} ACH")
        ax.plot(c, hm, col + "-", lw=1.3, label=f"{lab} ADoNIS")
        g = ho > 0.05 * ho.max()
        axr.plot(c[g], hm[g] / ho[g], col + mk, ms=3)
        x2, nd = chi2_ndf(hm, np.zeros_like(hm), ho, eo, floor=0.05)
        chis[lab] = x2 / max(nd, 1)
    ax.set_ylabel("norm. / bin"); ax.set_xticklabels([])
    axr.axhline(1, ls="--", color="gray"); axr.set_ylabel("ADoNIS/ACH"); axr.set_xlabel(xlabel)
    axr.set_ylim(0.5, 1.5)
    return chis


def fb(o, j):
    c = o[:, 0]; h = o[:, 1 + 2 * j]
    return (h[c > 0].sum() - h[c < 0].sum()) / h.sum()


def meanW(o, j):
    return np.average(o[:, 0], weights=o[:, 1 + 2 * j])


fig, axes = plt.subplots(2, 2, figsize=(12, 7), height_ratios=[3, 1])
ch_c = panel(axes[0, 0], axes[1, 0], oc[0], 0, 16, (-1, 1), r"$\cos\theta^*_\pi$")
axes[0, 0].set_title(r"pion CM angle $\cos\theta^*_\pi$  (A_FB e=%+.3f, ν$_e$=%+.3f, ν$_μ$=%+.3f)"
                     % (fb(oc[0], 0), fb(oc[0], 1), fb(oc[0], 2)))
axes[0, 0].legend(fontsize=7, ncol=3,
                  title="χ²/ndf  " + "  ".join(f"{k}:{v:.1f}" for k, v in ch_c.items()))
ch_w = panel(axes[0, 1], axes[1, 1], oc[1], 1, 20, (1080, 1700), "W [MeV]")
axes[0, 1].set_title(r"W  (⟨W⟩ e=%.0f, ν$_e$=%.0f, ν$_μ$=%.0f)"
                     % (meanW(oc[1], 0), meanW(oc[1], 1), meanW(oc[1], 2)))
axes[0, 1].legend(fontsize=7, ncol=3,
                  title="χ²/ndf  " + "  ".join(f"{k}:{v:.1f}" for k, v in ch_w.items()))

fig.suptitle(r"Exclusive e vs ν$_e$ vs ν$_μ$ single-π on $^{12}$C, matched kinematics "
             r"(E=2.222 GeV, $\theta_\ell\in[14,17]°$) — ADoNIS vs ACHILLES RES")
fig.tight_layout()
out = ROOT / "figures"; out.mkdir(exist_ok=True)
fig.savefig(out / "exclusive_evnu_c12.png", dpi=130)
print("wrote", out / "exclusive_evnu_c12.png")
print("A_FB (oracle):  e %+.3f  nu_e %+.3f  nu_mu %+.3f"
      % (fb(oc[0], 0), fb(oc[0], 1), fb(oc[0], 2)))
print("<W> (oracle):  e %.0f  nu_e %.0f  nu_mu %.0f MeV"
      % (meanW(oc[1], 0), meanW(oc[1], 1), meanW(oc[1], 2)))
print("chi2/ndf cos*:", {k: round(v, 1) for k, v in ch_c.items()})
print("chi2/ndf W:   ", {k: round(v, 1) for k, v in ch_w.items()})
