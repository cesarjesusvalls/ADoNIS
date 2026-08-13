"""Run the section-5 unfolding studies and persist everything the figures need.

One pass: build the inputs from the bank, run every study, write output/altgen/<label>_unfold.npz.
The figures never refit -- same rule as section 4, so restyling costs seconds and a figure can never
disagree with the fit it claims to show.

Studies:
  asimov       data at the generator's own truth.  Everything must come back at 1.
  sig120       signal scaled by 1.20.  The templates must find it and nothing else may move.
  flux120      the FLUX peak scaled by 1.20 instead.  This is the separability test: flux and templates
               both scale the signal, so if the fit cannot tell them apart it will report a 20% signal
               excess that is not there.
  budget       the same Asimov fit with systematic blocks switched on one at a time.
"""
from __future__ import annotations

import numpy as np

from adonis.analysis import knobs as K
from adonis.analysis.sample import AnaSample
from adonis.unfold import flux as FX
from adonis.unfold.binning import reco_grid, truth_grid
from adonis.unfold.build import build
from adonis.unfold.fit import UnfoldEngine

BUDGET = [("stat", dict(free_flux=False, free_det=False, use_knobs=False)),
          ("xsec", dict(free_flux=False, free_det=False, use_knobs=True)),
          ("flux", dict(free_flux=True, free_det=False, use_knobs=True)),
          ("det", dict(free_flux=True, free_det=True, use_knobs=True))]


def main(sample="configs/samples/t2k_cc0pi.yaml", label="sec5", norm=50_000, max_chunks=None,
         threshold=1.0, log=print):
    s = AnaSample.from_config(sample)
    inp = build(s.bank, s.cfg.signal, norm_events=norm, max_chunks=max_chunks, log=log)
    eng = UnfoldEngine(inp)
    TG, RG = truth_grid(), reco_grid()
    d0, s0 = eng.asimov()
    idx, imp = eng.select_dials(s0, threshold=threshold, log=log)

    out = dict(A=inp["A"], n_true=inp["n_true"], eff=inp["eff"], purity=inp["purity"],
               n_sig_reco=inp["n_sig_reco"], n_bkg_reco=inp["n_bkg_reco"],
               dial_impact=imp, dial_names=np.array(K.PNAMES), dial_idx=idx,
               true_dpt=TG.dpt, true_dat=TG.dat, reco_dpt=RG.dpt, reco_dat=RG.dat,
               flux_edges=np.array(FX.FLUX_EDGES), flux_cov=FX.prior_cov(),
               det_prior=eng.det_prior, nflux=eng.nflux)

    # FLUX INJECTIONS.  Flux and templates both scale the signal, so these are the tests that decide
    # whether the two blocks are separable at all: a fit that cannot tell them apart will report a
    # signal excess that is not there, or absorb a real one into the flux.
    f_peak = np.ones(eng.nflux); f_peak[1:5] = 1.20               # peak bins only, 600-1000 MeV
    f_tilt = np.linspace(0.85, 1.15, eng.nflux)                   # a SHAPE change, mean ~1
    studies = {"asimov":  dict(c=np.ones(TG.n)),
               "sig120":  dict(c=np.full(TG.n, 1.20)),
               "flux120": dict(c=np.ones(TG.n), f=f_peak),
               "fluxtilt": dict(c=np.ones(TG.n), f=f_tilt),
               # the hard one: signal and flux distorted at the same time, in different variables
               "combo":   dict(c=np.full(TG.n, 1.15), f=f_tilt)}
    for name, tru in studies.items():
        ct = tru["c"]; ft = tru.get("f")
        data, sig = eng.asimov(c_true=ct, f_true=ft)
        r = eng.fit(data, sig, knob_idx=idx, log=log)
        out[f"{name}_c"] = r["c"]; out[f"{name}_c_err"] = r["c_err"]
        out[f"{name}_f"] = r["f"]; out[f"{name}_f_err"] = r["f_err"]
        out[f"{name}_det"] = r["det"]; out[f"{name}_det_err"] = r["det_err"]
        out[f"{name}_c_true"] = ct
        out[f"{name}_f_true"] = np.ones(eng.nflux) if ft is None else ft
        out[f"{name}_cov"] = r["cov"]; out[f"{name}_chi2"] = r["chi2"]
        out[f"{name}_data"] = data; out[f"{name}_sigma"] = sig
        if log:
            log(f"[{name}] chi2 {r['chi2']:.2e}  max|c-truth| {np.abs(r['c'] - ct).max():.2e}")

    data, sig = eng.asimov()
    for name, kw in BUDGET:
        ki = idx if kw.pop("use_knobs") else np.zeros(0, int)
        r = eng.fit(data, sig, knob_idx=ki, log=None, **kw)
        out[f"budget_{name}"] = r["c_err"]
        if log:
            log(f"[budget] {name:5s} mean c_err {r['c_err'].mean():.4f}")

    path = f"output/altgen/{label}_unfold.npz"
    np.savez(path, **out)
    if log:
        log(f"[out] {path}")
    return path


if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:3] or []))
