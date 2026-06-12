"""Figure: CC0pi-Np (T2K signal) ABSOLUTE dsigma/d(delta_alphaT) and dsigma/d(delta_pT), ADoNIS
(CH, fixed cascade) vs ACHILLES, with ACH/ADO ratio panels and the T2K STV data overlaid.
First-principles absolute nb, no fit.  Shared files: cc0pi_disaggregated.npz, t2k_cc0pi_tki_achilles.npz,
t2k_cc0pi_stv_data.npz.

T2K data (arXiv:1802.05078) is dsigma/dx per NUCLEON in cm^2; converted to per-12C nb via
x 1e33 (nb/cm^2) x 12 (nucleons).  Integral check (printed) confirms consistency with the generators."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# optional tuned overlay: --theta SABS SSCAT MA  (reweights the stored walk/M_A records --
# no event regeneration; requires an npz produced with records, 2026-06-12+)
THETA = None
if "--theta" in sys.argv:
    i = sys.argv.index("--theta")
    THETA = tuple(float(x) for x in sys.argv[i + 1:i + 4])

ORA = Path("data/oracle")
dis = np.load(ORA / "cc0pi_disaggregated.npz")
ach = np.load(ORA / "t2k_cc0pi_tki_achilles.npz")
dat = np.load(ORA / "t2k_cc0pi_stv_data.npz")
SCALE = 6.266697e-02 * 1e-3 / 8.905269e+05
NB_PER_CM2, A_NUCLEON = 1e33, 12.0                              # per-nucleon data -> per-12C nb
CELLS = ["QE-C", "RES-C", "RES-H"]
ado = {k: np.concatenate([dis[f"{c}_True_{k}"] for c in CELLS]) for k in ("dalphat", "dpt", "w")}
ach_w = np.asarray(ach["w"]) * SCALE


def tuned_weights(theta):
    """Per-event tuned weights w(theta) for the concatenated CELLS, from the stored records
    (kind-1 reweight: pion branch x nucleon sigma_scatter x exact M_A quadratic)."""
    from adonis.analysis.ma_records import ma_reweight
    from adonis.fsi.cascade_discrete import nucleon_scat_reweight, pion_branch_reweight
    import jax.numpy as jnp
    sabs, sscat, MA = theta
    out = []
    for c in CELLS:
        w = np.asarray(dis[f"{c}_True_w"], float).copy()
        if f"{c}_True_rec_ma_a" not in dis.files or len(w) == 0:
            out.append(w)                                          # RES-H: zero CC0pi, no records
            continue
        g = lambda k: dis[f"{c}_True_rec_{k}"]
        wma = np.asarray(ma_reweight((g("ma_a"), g("ma_b"), g("ma_c"), g("ma_q2")), MA))
        ws = np.asarray(nucleon_scat_reweight((jnp.asarray(g("s1_hh")), jnp.asarray(g("s1_a")),
                                               jnp.asarray(g("s1_ns"))), sscat))
        ws2 = np.asarray(nucleon_scat_reweight((jnp.asarray(g("s2_hh")), jnp.asarray(g("s2_a")),
                                                jnp.asarray(g("s2_ns"))), sscat))
        ws = ws * np.where(np.asarray(g("has_ko")), ws2, 1.0)
        wb = 1.0
        if f"{c}_True_rec_b_bc" in dis.files:                      # pion cascade (RES only)
            wb = np.asarray(pion_branch_reweight((jnp.asarray(g("b_bc")), jnp.asarray(g("b_sa")),
                                                  jnp.asarray(g("b_ss")), jnp.asarray(g("b_si")),
                                                  jnp.asarray(g("b_nh"))), sabs, sscat))
        out.append(w * wma * ws * wb)
    return np.concatenate(out)


ado_w_tuned = tuned_weights(THETA) if THETA is not None else None

# (key, T2K data edges [native], x-scale to ADoNIS units, label, data unit-scale to nb/[ADoNIS unit])
VARS = [("dalphat", dat["dalphat_edges"], 1.0, r"$\delta\alpha_T$ [rad]",
         NB_PER_CM2 * A_NUCLEON, dat["dalphat_val"], dat["dalphat_err"]),
        ("dpt", dat["dpt_edges"] * 1000.0, 1.0, r"$\delta p_T$ [MeV]",
         NB_PER_CM2 * A_NUCLEON / 1000.0, dat["dpt_val"], dat["dpt_err"])]


def hist(v, w, bins):
    v = np.asarray(v); m = np.isfinite(v) & np.isfinite(w)
    h, _ = np.histogram(v[m], bins=bins, weights=w[m]); h2, _ = np.histogram(v[m], bins=bins, weights=w[m] ** 2)
    return h, np.sqrt(h2)


fig, axes = plt.subplots(2, len(VARS), figsize=(6.5 * len(VARS), 7.5), height_ratios=[3, 1], sharex="col")
for c, (key, bins, _xs, xlab, dscale, dval, derr) in enumerate(VARS):
    bw = np.diff(bins); ctr = 0.5 * (bins[1:] + bins[:-1])
    da, ea = hist(ach[key], ach_w, bins); dd, ed = hist(ado[key], ado["w"], bins)
    da, ea, dd, ed = da / bw, ea / bw, dd / bw, ed / bw          # dsigma/dx [nb/unit]
    d_y = dval * dscale; d_e = derr * dscale                     # T2K data -> nb/[ADoNIS unit], per-12C
    ax, axr = axes[0, c], axes[1, c]
    ax.step(bins, np.append(da, da[-1]), where="post", color="0.35", lw=1.5, label="ACHILLES")
    ax.errorbar(ctr, da, yerr=ea, fmt="none", ecolor="0.35", alpha=0.5)
    ax.step(bins, np.append(dd, dd[-1]), where="post", color="C0", lw=1.5, label="ADoNIS (CH, fixed)")
    ax.errorbar(ctr, dd, yerr=ed, fmt="none", ecolor="C0", alpha=0.5)
    if ado_w_tuned is not None:
        dt_, et_ = hist(ado[key], ado_w_tuned, bins); dt_, et_ = dt_ / bw, et_ / bw
        ax.step(bins, np.append(dt_, dt_[-1]), where="post", color="C1", lw=1.6, ls="--",
                label=f"ADoNIS tuned ($\\sigma_a$={THETA[0]:g}, $\\sigma_s$={THETA[1]:g}, $M_A$={THETA[2]:g})")
    ax.errorbar(ctr, d_y, yerr=d_e, fmt="o", color="k", ms=5, capsize=3, lw=1.4, label="T2K data", zorder=5)
    ax.set_ylabel(r"d$\sigma$/dx [nb/unit]"); ax.set_ylim(bottom=0); ax.legend()
    ax.set_title(f"CC0$\\pi$-Np   {xlab}")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = da / dd; re = r * np.sqrt((ed / dd) ** 2 + (ea / da) ** 2)
    msk = (da > 0) & (dd > 0) & np.isfinite(re)
    chi2 = float(np.sum((da[msk] - dd[msk]) ** 2 / (ea[msk] ** 2 + ed[msk] ** 2))); ndf = int(msk.sum())
    axr.axhspan(0.9, 1.1, color="green", alpha=0.12); axr.axhline(1.0, ls="--", color="green")
    axr.errorbar(ctr[msk], r[msk], yerr=re[msk], fmt="o", color="C3", ms=4, capsize=2, lw=1.1)
    axr.set_ylim(0.6, 1.4); axr.set_ylabel("ACHILLES / ADoNIS"); axr.set_xlabel(xlab)
    axr.text(0.03, 0.85, f"$\\chi^2$/ndf = {chi2/max(ndf,1):.2f}", transform=axr.transAxes, fontsize=10)
    print(f"  {key}: integral nb  ACH={np.sum(da*bw):.3e}  ADO={np.sum(dd*bw):.3e}  T2K={np.sum(d_y*bw):.3e}")
fig.suptitle("CC0$\\pi$-Np (T2K signal) absolute d$\\sigma$/dx — ADoNIS vs ACHILLES vs T2K data "
             "(arXiv:1802.05078)", fontsize=13)
fig.tight_layout()
out = "paper_figures/cc0pi_dat_dpt_data.png" if THETA is None else "paper_figures/cc0pi_dat_dpt_data_tuned.png"
fig.savefig(out, dpi=120); print("wrote", out)
