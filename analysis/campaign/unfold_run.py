"""Run the unfolding studies and persist everything the figures need.

One pass: build the inputs from the bank, run every study, write <results>/<label>_unfold.npz.  The
figures never refit, so restyling costs seconds and a figure can never disagree with the fit it shows.

Studies:
  asimov       data at the generator's own truth.  Everything must come back at 1.
  sig120       signal scaled by 1.20.  The templates must find it and nothing else may move.
  flux120      the FLUX peak scaled by 1.20 instead: the separability test, since flux and templates both
               scale the signal.
  budget       the same Asimov fit with systematic blocks switched on one at a time.
"""
from __future__ import annotations

from analysis._cli import results_dir

import numpy as np

from adonis.reweight import knobs as K
from analysis.campaign.sample import AnaSample
from adonis.unfold import flux as FX
from adonis.unfold.binning import StaircaseGrid
from adonis.unfold.config import UnfoldConfig


def _grid_arrays(G, tag):
    """Persist a grid's geometry whatever its type: two edge arrays for a rectangular grid, or a
    per-slice edge list plus cos boundaries (object array, rows of different lengths) for a staircase.
    """
    if isinstance(G, StaircaseGrid):
        pl, ph, cl, ch = G.cell_bounds()
        return {f"{tag}_kind": "staircase", f"{tag}_cos_edges": G.cos_edges,
                f"{tag}_p_lo": pl, f"{tag}_p_hi": ph, f"{tag}_c_lo": cl, f"{tag}_c_hi": ch,
                f"{tag}_slice_counts": np.array(G.counts)}
    return {f"{tag}_kind": "rect", f"{tag}_dpt": G.dpt, f"{tag}_dat": G.dat}
from adonis.unfold.build import build
from adonis.unfold.fit import UnfoldEngine

_BUDGET_ENABLE = {"stat": {}, "xsec": {"use_knobs": True},
                  "flux": {"free_flux": True}, "det": {"free_det": True}}


def _budget_sequence(names):
    """(name, kwargs) per entry, cumulative, with a FRESH dict each time."""
    state = dict(free_flux=False, free_det=False, use_knobs=False)
    for n in names:
        if n not in _BUDGET_ENABLE:
            raise ValueError(f"budget: unknown block {n!r}; allowed {sorted(_BUDGET_ENABLE)}")
        state = {**state, **_BUDGET_ENABLE[n]}
        yield n, dict(state)


def main(config="configs/fits/unfold.yaml", label=None, log=print):
    """Run every study in `config` and persist one npz.  Nothing here reads the environment."""
    cfg = UnfoldConfig.load(config)
    label = label or cfg.name
    s = AnaSample.from_config(cfg.sample_path())
    spec = cfg.smear_spec()
    TG, RG = cfg.grids()
    inp = build(s.bank, s.cfg.signal, spec=spec, norm_events=cfg.norm_events,
                max_chunks=cfg.max_chunks, log=log, obs_mode=cfg.binning.observables, TG=TG, RG=RG)
    _edges = cfg.flux.edges if cfg.flux.edges else None
    eng = UnfoldEngine(inp, knob_prior=K.PRIOR, det_prior=cfg.priors.detector, flux_edges=_edges,
                       flux_sigma=cfg.flux.sigma, flux_corr=cfg.flux.corr_length)
    d0, s0 = eng.asimov()
    idx, imp = eng.select_dials(s0, threshold=cfg.priors.dial_threshold, log=log)

    out = dict(A=inp["A"], n_true=inp["n_true"], eff=inp["eff"], purity=inp["purity"],
               norm_scale=inp["scale"], w_total_raw=inp["w_total_raw"], norm_events=inp["norm_events"],
               n_sig_reco=inp["n_sig_reco"], n_bkg_reco=inp["n_bkg_reco"],
               dial_impact=imp, dial_names=np.array(K.PNAMES), dial_idx=idx,
               obs_mode=cfg.binning.observables, sigma_p=spec.sigma_p,
               sigma_theta_deg=spec.sigma_theta_deg, pi_eff=spec.pi_eff,
               pi_eff_p_max=spec.pi_eff_p_max, cfg_json=cfg.as_json(), cfg_name=cfg.name,
               **_grid_arrays(TG, "true"), **_grid_arrays(RG, "reco"),
               flux_edges=np.array(_edges if _edges else FX.FLUX_EDGES),
               flux_cov=FX.prior_cov(cfg.flux.sigma, cfg.flux.corr_length, _edges),
               det_prior=eng.det_prior, nflux=eng.nflux)

    pri = np.concatenate([np.full(TG.n, np.inf), np.full(eng.nflux, eng.flux_sigma),
                          eng.prior[idx], np.full(eng.nreco, eng.det_prior)])
    out["param_prior"] = pri
    out["param_block"] = np.array(["template"] * TG.n + ["flux"] * eng.nflux
                                  + ["xsec"] * len(idx) + ["detector"] * eng.nreco)
    out["param_name"] = np.array([f"c[{j}]" for j in range(TG.n)]
                                 + [f"f[{b}]" for b in range(eng.nflux)]
                                 + [K.PNAMES[k] for k in idx]
                                 + [f"d[{i}]" for i in range(eng.nreco)])

    for st in cfg.studies:
        ct = np.full(TG.n, st.scale_c)
        ft = None
        if st.scale_flux_peak != 1.0 or st.flux_tilt:
            ft = np.ones(eng.nflux)
            if st.scale_flux_peak != 1.0:
                ft[1:5] = st.scale_flux_peak
            if st.flux_tilt:
                ft = ft * np.linspace(1.0 - st.flux_tilt, 1.0 + st.flux_tilt, eng.nflux)
        data, sig = eng.asimov(c_true=ct, f_true=ft)
        r = eng.fit(data, sig, knob_idx=idx, log=log)
        name = st.name
        out[f"{name}_c"] = r["c"]; out[f"{name}_c_err"] = r["c_err"]
        out[f"{name}_f"] = r["f"]; out[f"{name}_f_err"] = r["f_err"]
        out[f"{name}_det"] = r["det"]; out[f"{name}_det_err"] = r["det_err"]
        out[f"{name}_th"] = r["th"]
        out[f"{name}_c_true"] = ct
        out[f"{name}_f_true"] = np.ones(eng.nflux) if ft is None else ft
        out[f"{name}_cov"] = r["cov"]; out[f"{name}_chi2"] = r["chi2"]
        out[f"{name}_data"] = data; out[f"{name}_sigma"] = sig
        if log:
            log(f"[{name}] chi2 {r['chi2']:.2e}  max|c-truth| {np.abs(r['c'] - ct).max():.2e}")

    data, sig = eng.asimov()
    for name, kw in _budget_sequence(cfg.budget):
        kw = dict(kw)
        ki = idx if kw.pop("use_knobs") else np.zeros(0, int)
        r = eng.fit(data, sig, knob_idx=ki, log=None, **kw)
        out[f"budget_{name}"] = r["c_err"]
        if log:
            log(f"[budget] {name:5s} mean c_err {r['c_err'].mean():.4f}")

    path = str(results_dir() / f"{label}_unfold.npz")
    np.savez(path, **out)
    if log:
        log(f"[out] {path}")
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run the unfolding studies from a config.")
    ap.add_argument("config", nargs="?", default="configs/fits/unfold.yaml")
    ap.add_argument("--label", default=None, help="output stem; defaults to the config's name")
    main(**vars(ap.parse_args()))
