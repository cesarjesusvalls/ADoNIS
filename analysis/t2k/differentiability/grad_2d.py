"""Per-knob 2D sensitivity heatmap over the (delta_pT, delta_alphaT) plane.

Build the model's joint distribution d2sigma/d(dpt,dat) (same frozen QE+RES proposal + pool walk + all the
per-event reweights as model_hist_full, but binned in 2D), then for EACH knob show its per-cell Jacobian
d(d2sigma)/dtheta as a red(+)/blue(-) diverging heatmap.  This is the arrow-plot concept in 2D: because a
cell is now an area, the up/down arrow becomes a color.  First panel = nominal d2sigma for orientation.
Empty (kinematically forbidden) cells are masked grey.  Per-panel symmetric color scale (own |J|max).

  python analysis/t2k/differentiability/grad_2d.py            # CC0PI_N env, NREP below
  python analysis/t2k/differentiability/grad_2d.py --plot-only /tmp/adonis_tune_runs/jac2d.npz
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

NREP = 4
_ISO = {0: "pp", 1: "pn", 2: "nn"}


def _specs(NOM):
    out = []
    for name, val in NOM.items():
        if name == "pw_norm":
            continue
        if isinstance(val, tuple):
            for i, vi in enumerate(val):
                lab = f"{name}[{_ISO[i]}]" if name.startswith("s_NN") else f"{name}[{i}]"
                out.append((name, i, lab, float(vi)))
        else:
            out.append((name, None, name, float(val)))
    return out


def _binning(obs):
    import uproot
    r = uproot.open(f"../nuisance/data/T2K/CC0pi/STV/{'dpt' if obs == 'dpt' else 'dat'}Results.root")
    e = np.asarray(r["Result"].axis().edges())
    return e * 1000.0 if obs == "dpt" else e               # dpt in MeV, dat in rad


def run():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    sys.argv = [sys.argv[0], "dpt"]
    from analysis.t2k.differentiability import tune as T
    from analysis.t2k.differentiability.full_knobs import nominal_knobs, build_hv_sf, _hv_qe, _hv_res, _fsi
    from adonis.analysis.sf_reweight import sf_reweight
    from adonis.workflow.materials import resolve_targets
    from adonis.xsec.spectral import SpectralFunction

    T.NQE = T.NRES = int(os.environ.get("CC0PI_N", "30000"))
    NOM = nominal_knobs()
    edpt, edat = _binning("dpt"), _binning("dat")
    n1, n2 = len(edpt) - 1, len(edat) - 1
    ej1, ej2 = jnp.asarray(edpt), jnp.asarray(edat)
    area = np.outer(np.diff(edpt), np.diff(edat))           # MeV * rad
    CONV2D = 1e-33 / 12.0 * 1e38 * 1000.0                   # -> 1e-38 cm^2/(GeV/c . rad)/nucleon
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    log(f"2D Jacobian heatmap (dpt x dat = {n1}x{n2})  N={T.NQE} NREP={NREP}")

    qe, qw, res, rw = T.build_proposal(); log("proposal sampled")
    sf = SpectralFunction(resolve_targets("C")[0][0].spectral_n)
    HV, SF = build_hv_sf(qe, res, sf, with_pw=False); log("hard-vertex amps2 records (no pw) + SF grids built")
    walks = [T.build_walk(jax.random.PRNGKey(50 + i), qe, qw, res, rw) for i in range(NREP)]
    log(f"{NREP} walk replicas built")

    def bin2d(W):
        qd, qa = T._dpt(W["q_kmu"], W["q_lead"]), T._dat(W["q_kmu"], W["q_lead"])
        rd, ra = T._dpt(W["r_kmu"], W["r_lead"]), T._dat(W["r_kmu"], W["r_lead"])
        qi = jnp.clip(jnp.searchsorted(ej1, qd) - 1, 0, n1 - 1); qj = jnp.clip(jnp.searchsorted(ej2, qa) - 1, 0, n2 - 1)
        ri = jnp.clip(jnp.searchsorted(ej1, rd) - 1, 0, n1 - 1); rj = jnp.clip(jnp.searchsorted(ej2, ra) - 1, 0, n2 - 1)
        return dict(q_flat=qi * n2 + qj, q_keep=T._sel(W["q_kmu"], W["q_lead"]), q_w0=W["q_w0"], q_rec=W["q_rec"],
                    r_flat=ri * n2 + rj, r_keep=T._sel(W["r_kmu"], W["r_lead"]), r_w0=W["r_w0"], r_rec=W["r_rec"])
    banks = [bin2d(W) for W in walks]

    qsf_args = (SF["grids"], SF["qe_pmag"], SF["qe_erem"]); rsf_args = (SF["grids"], SF["res_pmag"], SF["res_erem"])
    tup_names = sorted({n for n, idx, _, _ in _specs(NOM) if idx is not None})
    SP = _specs(NOM); labels = [s[2] for s in SP]; p0 = jnp.asarray([s[3] for s in SP])

    def assemble(p):
        k = dict(NOM); tup = {n: list(NOM[n]) for n in tup_names}
        for i, (name, idx, _, _) in enumerate(SP):
            (k.__setitem__(name, p[i]) if idx is None else tup[name].__setitem__(idx, p[i]))
        for n in tup_names:
            k[n] = tuple(tup[n])
        return k

    def hist2d_flat(p):
        k = assemble(p)
        qsf = sf_reweight(*qsf_args, kF_sf=k["kF_sf"], Eb_shift=k["Eb_shift"], sf_norm=k["sf_norm"], src_tail=k["src_tail"])
        rsf = sf_reweight(*rsf_args, kF_sf=k["kF_sf"], Eb_shift=k["Eb_shift"], sf_norm=k["sf_norm"], src_tail=k["src_tail"])
        hvq, hvr = _hv_qe(k, HV), _hv_res(k, HV)
        acc = jnp.zeros(n1 * n2)
        for R in banks:
            q_w = R["q_w0"] * k["qe_norm"] * hvq * _fsi(R["q_rec"], k) * qsf
            r_w = R["r_w0"] * k["res_norm"] * hvr * _fsi(R["r_rec"], k) * rsf
            acc = acc + (jax.ops.segment_sum(q_w * R["q_keep"], R["q_flat"], num_segments=n1 * n2)
                         + jax.ops.segment_sum(r_w * R["r_keep"], R["r_flat"], num_segments=n1 * n2))
        return acc / len(banks)

    h0 = np.asarray(hist2d_flat(p0)).reshape(n1, n2)
    Jflat = np.asarray(jax.jacrev(hist2d_flat)(p0))                  # (n1*n2, nknob)
    log(f"2D Jacobian done: {Jflat.shape[1]} knobs x {n1*n2} cells")
    norm2d = (h0 / area) * CONV2D
    J2d = (Jflat.T.reshape(len(labels), n1, n2) / area[None]) * CONV2D    # (nknob, n1, n2)

    os.makedirs("/tmp/adonis_tune_runs", exist_ok=True)
    np.savez("/tmp/adonis_tune_runs/jac2d.npz", edpt=edpt, edat=edat, norm2d=norm2d, J2d=J2d,
             labels=np.array(labels))
    make_figure("/tmp/adonis_tune_runs/jac2d.npz")


def make_figure(npz_path, relative=False, rate_frac=0.01):
    """Per-knob 2D sensitivity heatmap.  relative=False -> ABSOLUTE d(d2sigma)/dtheta; relative=True ->
    FRACTIONAL d(ln sigma)/dtheta = J2d/norm2d (cells below rate_frac*max(norm2d) masked: relative is just
    noise where there is ~no rate).  Per-panel symmetric scale (robust 99th pct for the relative case)."""
    d = np.load(npz_path, allow_pickle=True)
    edpt, edat = d["edpt"] / 1000.0, d["edat"]              # dpt back to GeV/c for axes
    norm2d, J2d, labels = d["norm2d"], d["J2d"], list(d["labels"])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if relative:
        mask = norm2d <= rate_frac * float(norm2d.max())    # drop low-rate cells (noisy ratio)
        denom = np.where(mask, 1.0, norm2d)
        field = J2d / denom[None]                            # d(ln sigma)/dtheta
        sym = "$\\partial(\\ln\\sigma)/\\partial\\theta$"; unit = "[/unit knob]"; tag = "relative"
        robust = True
    else:
        mask = norm2d <= 0                                   # kinematically empty cells
        field = J2d
        sym = "$\\partial(\\mathrm{d}^2\\sigma)/\\partial\\theta$"; unit = ""; tag = "absolute"
        robust = False
    n = len(labels); ncol = 6; nrow = int(np.ceil((n + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.7 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    X, Y = np.meshgrid(edpt, edat, indexing="ij")

    nm = np.ma.masked_where(mask, norm2d)
    a0 = axes[0]
    pc = a0.pcolormesh(X, Y, nm, cmap="Greys", shading="flat")
    a0.set_title("nominal d$^2\\sigma$/d$\\delta p_T$d$\\delta\\alpha_T$", fontsize=8)
    a0.set_ylabel(r"$\delta\alpha_T$ [rad]", fontsize=8); fig.colorbar(pc, ax=a0, fraction=0.046)

    for j, lab in enumerate(labels):
        ax = axes[j + 1]; F = np.ma.masked_where(mask, field[j])
        vals = np.abs(field[j][~mask])
        vmax = float(np.percentile(vals, 99) if robust and vals.size else (np.max(vals) if vals.size else 1.0)) or 1.0
        pc = ax.pcolormesh(X, Y, F, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
        ax.set_title(f"{lab}  max={vmax:.1e}", fontsize=8)
        fig.colorbar(pc, ax=ax, fraction=0.046)
    for j in range(n + 1, len(axes)):
        axes[j].axis("off")
    for k in range(ncol):
        idx = (nrow - 1) * ncol + k
        if idx < n + 1:
            axes[idx].set_xlabel(r"$\delta p_T$ [GeV/c]", fontsize=8)
    for r in range(nrow):
        axes[r * ncol].set_ylabel(r"$\delta\alpha_T$ [rad]", fontsize=8)
    extra = f"; cells <{rate_frac:.0%} of peak rate masked" if relative else ""
    fig.suptitle(f"Per-knob 2D {tag} sensitivity {sym} {unit} over "
                 f"$(\\delta p_T,\\,\\delta\\alpha_T)$  (red=+ / blue=$-$; per-panel symmetric scale{extra})",
                 fontsize=12)
    os.makedirs("output/figures", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    stem = "cc0pi_jacobian_2d_relative" if relative else "cc0pi_jacobian_2d"
    for ext in ("png", "pdf"):
        Fn = f"output/figures/{stem}.{ext}"; fig.savefig(Fn, dpi=120); print(f"wrote {Fn}", flush=True)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        make_figure(sys.argv[sys.argv.index("--plot-only") + 1], relative="--relative" in sys.argv)
    else:
        run()
