"""Assemble output/altgen/multisample_carbon.npz -- the shared sec2 (Fisher) + sec3 (gradient) input --
from the AnaSample samples + the FSI beam Jacobians.  A THIN caller: it composes the samples through the
core object (adonis.analysis.sample.SampleSet) and stacks the cached beam jvps; no selection/binning lives
here.  This replaces sec3_gradients/build_multisample.py (the per-sample physfit_*.npz drivers are gone).

    python -m analysis.paper.gate1_multisample                       # full banks (a GPU/big-node job)
    ADONIS_MS_MAXCHUNKS=4 python -m analysis.paper.gate1_multisample # smoke (subsampled banks)

dskeys are namespaced `sample:obs` (t2k_cc0pi:dpt) + the beam keys (pip_react, ...); sec2/sec3 read them.
The CH T2K CC1pi+ sample (t2k_cc1pi_ch) is used -- the non-pure-target demonstration.
"""
import os
import sys
import pathlib
import time

import yaml
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adonis.analysis.sample import SampleSet, gate1_from     # noqa: E402
from adonis.analysis import knobs as K                        # noqa: E402
from analysis.paper.beams import beam_fisher as BF            # noqa: E402
from analysis.paper import style                              # noqa: E402

# minerva_cc1pip_{tpi,q2} (arXiv:2605.24224) add the RES Q2 lever arm: the RES axial block enters as
# dipole(Q2; M_A_res) * res_axial_strength, so M_A_res (Q2 SHAPE) and C5A (NORMALISATION) are only
# separable with Q2 reach.  NOTE the two are the SAME 91,843 events binned two ways -- stacking both into
# one Fisher double-counts them; keep that in mind when reading the combined Gate I.
# The sample list and the beam observable keys are NOT declared here.  They come from the fit config
# (configs/fits/*.yaml) and configs/samples/beams.yaml respectively, because this Jacobian and the sec4
# engine must describe the SAME stack -- multisample.py asserts its dskeys against this npz, and when the
# two lists lived in two files, adding MINERvA CC1pi+ meant editing both in step or getting an assertion
# hours into a run.  BEAM_OBS was additionally duplicated in sec3_gradients/build_multisample.py.
_BEAMS_CFG = "configs/samples/beams.yaml"


def _beam_spec():
    d = yaml.safe_load(pathlib.Path(_BEAMS_CFG).read_text())
    return d["beams"], int(d["nbins"]), float(d["syst"])


def build(fit_config="configs/fits/sec4_P1.yaml", samples=None, beams=None, nbins=None, syst=None,
          max_chunks=None, out_label="multisample_carbon"):
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    from adonis.fit.config import FitConfig
    cfg = FitConfig.load(fit_config)
    bspec, nb_cfg, syst_cfg = _beam_spec()
    samples = list(samples if samples is not None else cfg.samples)
    beams = list(beams if beams is not None else cfg.beams)
    nbins = nb_cfg if nbins is None else nbins
    syst = syst_cfg if syst is None else syst
    log(f"from {fit_config}: {len(samples)} samples + {len(beams)} beams; beams from {_BEAMS_CFG}")

    ss = SampleSet.from_configs([f"configs/samples/{s}.yaml" for s in samples])
    r = ss.gate1(max_chunks=max_chunks, log=log)                         # the 5 experiment samples
    J = [r["J"]]; sigma = [r["sigma"]]; dskeys = list(r["keys"]); row0 = list(r["row0"])
    edges = {f"{k}_edges": r["edges"][k] for k in r["keys"]}
    central = {f"{k}_central": r["central"][k] for k in r["keys"]}

    for beam in beams:                                                   # FSI-only beam Jacobians (cached)
        Jb, sb, _c, e_beam, _n = BF.beam_jacobian(beam, nbins=nbins, syst=syst)
        J.append(np.asarray(Jb)); sigma.append(np.asarray(sb))
        for key in bspec[beam]["observables"]:
            dskeys.append(key); row0.append(row0[-1] + nbins); edges[f"{key}_edges"] = np.asarray(e_beam)
        log(f"  + beam {beam} ({np.asarray(Jb).shape[0]} bins)")

    J = np.vstack(J); sigma = np.concatenate(sigma); row0 = np.asarray(row0)
    F, V, _sig_post, shrink, _reach = gate1_from(J, sigma, K.PRIOR)      # joint Gate I on the FULL stack

    style.ALTGEN.mkdir(parents=True, exist_ok=True)
    out = style.ALTGEN / f"{out_label}.npz"
    np.savez(out, J=J, sigma=sigma, prior=K.PRIOR, pnames=K.PNAMES, dskeys=dskeys, row0=row0,
             shrink=shrink, F=F, V=V, **edges, **central)
    log(f"[out] {out}")
    log(f"  bins={J.shape[0]}  knobs={J.shape[1]}  observables={len(dskeys)}  "
        f"({'+'.join(list(samples) + list(beams))})")
    log(f"  {int((shrink < 0.5).sum())}/{len(K.PRIOR)} knobs pass Gate I on the combined set")
    return out


if __name__ == "__main__":
    mc = os.environ.get("ADONIS_MS_MAXCHUNKS")
    build(max_chunks=int(mc) if mc else None)
