"""Assemble the shared Fisher/shrinkage + per-bin gradient input from the AnaSample samples and the
FSI beam Jacobians.  A thin caller: composes the samples through analysis.campaign.sample.SampleSet
and stacks the cached beam jvps; no selection/binning lives here.

    python -m analysis.campaign.gate1 --label multisample_carbon
    python -m analysis.campaign.gate1 --label smoke --max-chunks 4

Writes <results>/<label>.npz.  dskeys are namespaced `sample:obs` (t2k_cc0pi:dpt) plus the beam
keys (pip_react, ...).
"""
import argparse
import os
import sys
import pathlib
import time

import yaml
from pathlib import Path

from analysis._cli import results_dir, timed_log

import numpy as np

ALTGEN = results_dir()

from analysis.campaign.sample import SampleSet, gate1_from
from adonis.reweight import knobs as K
from analysis.campaign import beams as BF

_BEAMS_CFG = "configs/samples/beams.yaml"


def _beam_spec():
    d = yaml.safe_load(pathlib.Path(_BEAMS_CFG).read_text())
    return d["beams"], int(d["nbins"]), float(d["syst"])


def build(fit_config="configs/fits/sec4_P1.yaml", samples=None, beams=None, nbins=None, syst=None,
          max_chunks=None, out_label="multisample_carbon"):
    log = timed_log()

    from adonis.fit.config import FitConfig
    cfg = FitConfig.load(fit_config)
    bspec, nb_cfg, syst_cfg = _beam_spec()
    samples = list(samples if samples is not None else cfg.samples)
    beams = list(beams if beams is not None else cfg.beams)
    nbins = nb_cfg if nbins is None else nbins
    syst = syst_cfg if syst is None else syst
    log(f"from {fit_config}: {len(samples)} samples + {len(beams)} beams; beams from {_BEAMS_CFG}")

    ss = SampleSet.from_configs([f"configs/samples/{s}.yaml" for s in samples])
    r = ss.gate1(max_chunks=max_chunks, log=log)
    J = [r["J"]]; sigma = [r["sigma"]]; dskeys = list(r["keys"]); row0 = list(r["row0"])
    edges = {f"{k}_edges": r["edges"][k] for k in r["keys"]}
    central = {f"{k}_central": r["central"][k] for k in r["keys"]}

    for beam in beams:
        Jb, sb, _c, e_beam, _n = BF.beam_jacobian(beam, nbins=nbins, syst=syst, max_chunks=max_chunks)
        J.append(np.asarray(Jb)); sigma.append(np.asarray(sb))
        for key in bspec[beam]["observables"]:
            dskeys.append(key); row0.append(row0[-1] + nbins); edges[f"{key}_edges"] = np.asarray(e_beam)
        log(f"  + beam {beam} ({np.asarray(Jb).shape[0]} bins)")

    J = np.vstack(J); sigma = np.concatenate(sigma); row0 = np.asarray(row0)
    F, V, _sig_post, shrink, _reach = gate1_from(J, sigma, K.PRIOR)

    ALTGEN.mkdir(parents=True, exist_ok=True)
    import json
    stamp = json.dumps({"fit_config": str(fit_config), "samples": list(samples),
                        "beams": list(beams), "nbins": nbins, "syst": syst,
                        "max_chunks": max_chunks}, sort_keys=True, default=str)
    out = ALTGEN / f"{out_label}.npz"
    np.savez(out, J=J, sigma=sigma, prior=K.PRIOR, pnames=K.PNAMES, dskeys=dskeys, row0=row0,
             cfg_json=stamp,
             shrink=shrink, F=F, V=V, **edges, **central)
    log(f"[out] {out}")
    log(f"  bins={J.shape[0]}  knobs={J.shape[1]}  observables={len(dskeys)}  "
        f"({'+'.join(list(samples) + list(beams))})")
    log(f"  {int((shrink < 0.5).sum())}/{len(K.PRIOR)} knobs pass Gate I on the combined set")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="analysis.campaign.gate1", description=__doc__.split("\n")[0])
    ap.add_argument("--label", default="multisample_carbon",
                    help="output name: <results>/<label>.npz")
    ap.add_argument("--fit-config", default="configs/fits/sec4_P1.yaml",
                    help="which config names the samples and beams")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="cap the bank chunks per sample (a smoke run; default: the whole bank)")
    a = ap.parse_args(argv)
    return build(fit_config=a.fit_config, max_chunks=a.max_chunks, out_label=a.label)


if __name__ == "__main__":
    main()
