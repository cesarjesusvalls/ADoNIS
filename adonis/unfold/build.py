"""Build the unfolding inputs from a bank: response, background, truth spectrum, efficiency, purity.

ONE streamed pass over the bank produces everything the fit needs.  For each chunk the SAME selection
code runs twice -- once on the true 4-vectors and once on the smeared ones -- so reco selection, reco
observables, efficiency and purity are all consequences of the detector rather than separate models:

    true_sel, true_obs = select_full(B,       sd)      # true signal phase space
    reco_sel, reco_obs = select_full(B_reco,  sd)      # what the "experiment" sees

Every reco-selected event is then one of two things, and the two are reweighted by DIFFERENT parameters:

  SIGNAL      true_sel and its TRUE (dpt, dat) lands in the truth grid.  Scaled by the template c_j of
              its TRUE bin.  Physics knobs do NOT touch it -- the template parameters are the only thing
              that moves the signal, which is what makes the unfolded result a measurement of the signal
              rather than of the model.
  BACKGROUND  everything else that survives the reco cuts: true non-signal, and true signal whose truth
              value falls outside the truth grid.  Reweighted by the 28 physics knobs, never by c.

so the prediction in reco bin i is

    mu_i(theta, c) = sum_j A_ij c_j  +  B_i(theta) ,    A_ij = sum over signal events (reco i, true j) w0

with A the (n_reco x n_truth) response.  A is a fixed array: the signal term is exactly linear in c, so
its Jacobian is A itself and needs no autodiff at all.

Efficiency and purity fall out of the same pass and are reported, not assumed:
    eff_j    = (signal events from truth bin j that pass reco) / (all true signal in truth bin j)
    purity_i = (signal in reco bin i) / (total in reco bin i)
"""
from __future__ import annotations

import glob

import numpy as np

from adonis.detector import SmearSpec, smear_chunk
from adonis.reweight import bank_plot as BP
from adonis.unfold.binning import reco_grid, truth_grid
from adonis.workflow import selection as SG


def _obs2(obs):
    """(dpt, dat) from a select_full observable dict, whatever the key spelling."""
    return np.asarray(obs["dpt"], dtype=float), np.asarray(obs["dalphat"], dtype=float)


def build(bank_dir, signal, spec: SmearSpec = None, norm_events=None, max_chunks=None, log=print,
          TG=None, RG=None):
    """Stream `bank_dir` and return the unfolding inputs.

    `norm_events`: scale every weight so the TOTAL PRE-SELECTION rate equals this (50 000 for section 5).
    Normalising by summed WEIGHT, never by row count -- the bank retains rejected events as dead rows
    carrying exactly w0 = 0 (0.9% of the T2K bank, some of which even pass the selection mask), so a
    count-based normalisation would silently include them and shift the scale.
    """
    spec = spec or SmearSpec()
    TG = TG or truth_grid()
    RG = RG or reco_grid()
    files = sorted(glob.glob(f"{bank_dir}/chunk_*.npz"))
    if max_chunks:
        files = files[:max_chunks]
    nch = BP.bank_nchunks(bank_dir)

    A = np.zeros((RG.n, TG.n))          # response: signal, reco bin x true bin
    n_true = np.zeros(TG.n)             # ALL true signal per truth bin (reco-selected or not) -> efficiency
    n_sig_reco = np.zeros(RG.n)         # signal that passes reco, per reco bin      -> purity numerator
    w_total = 0.0                       # total pre-selection rate (the 50k normalisation target)

    # BACKGROUND is kept as a COMPACT BANK, not as (chunk, index) pointers.  Its weight has to be
    # recomputed from the hard-vertex records at every fit iteration, so the records must stay resident;
    # pointers would mean re-reading the bank per iteration.  filter_events guarantees
    # bank_weight(filter_events(B, m), theta) == bank_weight(B, theta)[m], so compacting changes nothing.
    bkg_parts, bkg_bins = [], []
    for ci, f in enumerate(files):
        B = BP.load_bank_chunk(f, nch)
        R = smear_chunk(B, spec, ci)
        w0 = np.asarray(B["w0"], dtype=float)
        w_total += w0.sum()

        tsel, tobs, _, _ = SG.select_full(B, signal)
        rsel, robs, _, _ = SG.select_full(R, signal)
        tdpt, tdat = _obs2(tobs)
        rdpt, rdat = _obs2(robs)
        tbin = TG.index(tdpt, tdat)
        rbin = RG.index(rdpt, rdat)

        # TRUE signal = passes the true selection AND lands inside the truth grid.  True signal outside
        # the grid has no template to scale it, so it is background by construction; calling it signal
        # would attach it to whichever c_j the clipping happened to choose.
        is_sig = tsel & (tbin >= 0)
        np.add.at(n_true, tbin[is_sig], w0[is_sig])

        keep = rsel & (rbin >= 0)
        sig_r = keep & is_sig
        np.add.at(A, (rbin[sig_r], tbin[sig_r]), w0[sig_r])
        np.add.at(n_sig_reco, rbin[sig_r], w0[sig_r])

        bkg_r = keep & ~is_sig
        if bkg_r.any():
            bkg_parts.append(BP.filter_events(B, bkg_r))
            bkg_bins.append(rbin[bkg_r].astype(np.int32))
        if log:
            log(f"  [unfold] chunk {ci + 1}/{len(files)}  reco-sel {int(keep.sum()):6d}"
                f"  sig {int(sig_r.sum()):6d}  bkg {int(bkg_r.sum()):6d}")
        del B, R

    bkg_bank = SG._concat_compact(bkg_parts)
    bkg_bin = np.concatenate(bkg_bins)
    n_bkg_reco = np.bincount(bkg_bin, weights=np.asarray(bkg_bank["w0"], float), minlength=RG.n)

    # ONE scale, applied to w0.  bank_weight is w0 * (knob factors), so scaling w0 carries the
    # normalisation into every theta-dependent background weight with no further bookkeeping.
    scale = 1.0 if norm_events is None else float(norm_events) / w_total
    A *= scale; n_true *= scale; n_sig_reco *= scale; n_bkg_reco *= scale
    bkg_bank["w0"] = np.asarray(bkg_bank["w0"], float) * scale

    with np.errstate(invalid="ignore", divide="ignore"):
        eff = np.where(n_true > 0, A.sum(axis=0) / n_true, np.nan)
        tot = n_sig_reco + n_bkg_reco
        pur = np.where(tot > 0, n_sig_reco / tot, np.nan)
    return dict(A=A, n_true=n_true, n_sig_reco=n_sig_reco, n_bkg_reco=n_bkg_reco,
                bkg_bank=bkg_bank, bkg_bin=bkg_bin,
                eff=eff, purity=pur, scale=scale, w_total_raw=w_total,
                norm_events=(w_total * scale))
