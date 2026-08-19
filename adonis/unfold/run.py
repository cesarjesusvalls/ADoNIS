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
from adonis.detector import SmearSpec
from adonis.unfold.binning import StaircaseGrid, reco_grid, truth_grid


def _grid_arrays(G, tag):
    """Persist a grid's geometry whatever its type, so the figures never re-derive it.

    A rectangular grid is two edge arrays; a staircase is a per-slice edge list plus the cos
    boundaries, stored as an object array because the rows have different lengths.
    """
    if isinstance(G, StaircaseGrid):
        pl, ph, cl, ch = G.cell_bounds()
        return {f"{tag}_kind": "staircase", f"{tag}_cos_edges": G.cos_edges,
                f"{tag}_p_lo": pl, f"{tag}_p_hi": ph, f"{tag}_c_lo": cl, f"{tag}_c_hi": ch,
                f"{tag}_slice_counts": np.array(G.counts)}
    return {f"{tag}_kind": "rect", f"{tag}_dpt": G.dpt, f"{tag}_dat": G.dat}
from adonis.unfold.build import build
from adonis.unfold.fit import UnfoldEngine

BUDGET = [("stat", dict(free_flux=False, free_det=False, use_knobs=False)),
          ("xsec", dict(free_flux=False, free_det=False, use_knobs=True)),
          ("flux", dict(free_flux=True, free_det=False, use_knobs=True)),
          ("det", dict(free_flux=True, free_det=True, use_knobs=True))]


def main(sample="configs/samples/t2k_cc0pi.yaml", label="sec5", norm=50_000, max_chunks=None,
         threshold=1.0, log=print):
    s = AnaSample.from_config(sample)
    # DETECTOR WORKING POINT.  The lepton-kinematics study runs a better detector than the STV one
    # (10% / 5 deg against 20% / 10 deg): the point of moving observables was that p_mu and
    # cos theta_mu are directly measured, and pairing them with the coarse STV detector would throw
    # that away.  Both numbers are stated in the output so a figure can never claim a resolution the
    # fit did not use.
    from adonis.unfold.binning import UNFOLD_OBS
    # pi_eff: 50% of charged pions below 400 MeV/c are missed.  Without it a topological CC0pi sample
    # has NO background (purity exactly 1.0), the knobs have nothing to reweight, and the cross-section
    # systematic is identically zero -- so the budget figure would report a term that is missing, not
    # small.  A missed pion turns a true CC1pi event into a reco CC0pi one, which is the dominant
    # background of the real measurement.
    spec = (SmearSpec(sigma_p=0.10, sigma_theta_deg=5.0, pi_eff_p_max=400.0, pi_eff=0.5)
            if UNFOLD_OBS == "lep" else SmearSpec())
    inp = build(s.bank, s.cfg.signal, spec=spec, norm_events=norm, max_chunks=max_chunks, log=log)
    eng = UnfoldEngine(inp)
    TG, RG = truth_grid(), reco_grid()
    d0, s0 = eng.asimov()
    idx, imp = eng.select_dials(s0, threshold=threshold, log=log)

    # PERSIST THE NORMALISATION.  n_true and A are scaled so the pre-selection sample is `norm` events,
    # which is a display choice, not physics.  Without `scale` stored, the figures cannot get back to
    # the raw summed w0 and therefore cannot express anything in cross-section units -- the absolute
    # scale would have to be re-derived from the bank, which is exactly how a plot ends up with a
    # normalisation nobody can trace.
    out = dict(A=inp["A"], n_true=inp["n_true"], eff=inp["eff"], purity=inp["purity"],
               norm_scale=inp["scale"], w_total_raw=inp["w_total_raw"], norm_events=inp["norm_events"],
               n_sig_reco=inp["n_sig_reco"], n_bkg_reco=inp["n_bkg_reco"],
               dial_impact=imp, dial_names=np.array(K.PNAMES), dial_idx=idx,
               obs_mode=UNFOLD_OBS, sigma_p=spec.sigma_p, sigma_theta_deg=spec.sigma_theta_deg,
               **_grid_arrays(TG, "true"), **_grid_arrays(RG, "reco"),
               flux_edges=np.array(FX.FLUX_EDGES), flux_cov=FX.prior_cov(),
               det_prior=eng.det_prior, nflux=eng.nflux)

    # PARAMETER LAYOUT of the fitted vector, so the covariance can be read without re-deriving it:
    # [ c (templates) | f (flux) | theta (cross section) | d (detector) ].  Template priors are infinite
    # by construction -- that is what "unconstrained" means, and it is why the flux, which IS penalised,
    # cannot move: the templates absorb the variation for free.
    pri = np.concatenate([np.full(TG.n, np.inf), np.full(eng.nflux, FX.SIGMA),
                          eng.prior[idx], np.full(eng.nreco, eng.det_prior)])
    out["param_prior"] = pri
    out["param_block"] = np.array(["template"] * TG.n + ["flux"] * eng.nflux
                                  + ["xsec"] * len(idx) + ["detector"] * eng.nreco)
    out["param_name"] = np.array([f"c[{j}]" for j in range(TG.n)]
                                 + [f"f[{b}]" for b in range(eng.nflux)]
                                 + [K.PNAMES[k] for k in idx]
                                 + [f"d[{i}]" for i in range(eng.nreco)])

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
        out[f"{name}_th"] = r["th"]
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
    # norm reachable from the CLI: it sets the assumed DATA exposure (not the MC), so it is the knob
    # for "what would this measurement look like with N times the statistics" and was previously
    # editable only by changing the default.
    a = sys.argv[1:4]
    main(*(a[:2] or []), **({"norm": int(float(a[2]))} if len(a) > 2 else {}))
