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
from adonis.unfold.flux import flux_index, n_flux
from scipy.special import erf
from adonis.workflow import selection as SG


def _obs2(obs):
    """The unfolded observable pair, per binning.UNFOLD_OBS.

    "stv" -> (delta-p_T [MeV], delta-alpha_T [rad])
    "lep" -> (p_mu [GeV/c], cos theta_mu).  p_mu is converted from the selection's MeV to the GeV/c of
             the published binning HERE and nowhere else, so there is exactly one place the unit can be
             wrong -- a factor 1000 in a momentum axis would move every event into the first bin and
             still produce a plausible-looking fit.
    """
    from adonis.unfold.binning import UNFOLD_OBS
    if UNFOLD_OBS == "lep":
        return (np.asarray(obs["pmu"], dtype=float) / 1000.0,
                np.asarray(obs["cos_mu"], dtype=float))
    return np.asarray(obs["dpt"], dtype=float), np.asarray(obs["dalphat"], dtype=float)


def _soft_membership(vals, edges, sigma):
    """Fraction of a Gaussian of width `sigma` centred on each value that falls in each bin.

    Separable in the two observables, so the 2-D cell membership is an outer product of two 1-D ones.
    An infinite top edge integrates to the tail, which is why the open delta-p_T bin still receives its
    share instead of silently losing it.
    """
    v = np.asarray(vals, float)[:, None]
    e = np.asarray(edges, float)[None, :]
    with np.errstate(invalid="ignore"):
        cdf = 0.5 * (1.0 + erf((e - v) / (sigma * np.sqrt(2.0))))
    cdf = np.where(np.isinf(e), np.where(e > 0, 1.0, 0.0), cdf)
    m = np.diff(cdf, axis=1)
    tot = m.sum(axis=1, keepdims=True)
    return np.divide(m, tot, out=np.zeros_like(m), where=tot > 0)


def build(bank_dir, signal, spec: SmearSpec = None, norm_events=None, max_chunks=None, log=print,
          TG=None, RG=None, soft_sigma=None, soft_reco=0):
    """Stream `bank_dir` and return the unfolding inputs.

    `soft_sigma`: (sigma_dpt, sigma_dat).  When given, a signal event does not go into ONE truth cell --
    its weight is spread over cells by a Gaussian of the detector's own resolution, so the templates
    become overlapping basis functions rather than disjoint indicators.  The motivation is that a basis
    finer than the resolution is not measurable, so matching the basis to what the detector can localise
    should condition better.  Note what it costs: c_j is then the coefficient of an overlapping basis
    function, NOT the rate in cell j, and the two are only comparable through a derived quantity.

    `soft_reco`: number of DETECTOR REPLICAS per event.  Instead of smearing once and putting the event
    in one reco bin, smear it `soft_reco` times and give each replica 1/N of the weight, so one event
    contributes to several reco bins in the proportion the detector actually produces.

    This resamples the REAL kernel rather than assuming one, which matters because the smearing is
    MULTIPLICATIVE: |p| -> |p|(1 + 0.2 g).  A +20% move and a -20% move are not mirror images (1/1.2 =
    0.833, not 0.8), the width scales with the value rather than being a constant, and the induced
    kernel in delta-p_T is skewed.  Any symmetric fixed-width approximation -- including the Gaussian
    used by `soft_sigma` -- gets that wrong; resampling cannot, because it never writes the kernel down.

    `norm_events`: scale every weight so the TOTAL PRE-SELECTION rate equals this (50 000 for section 5).
    Normalising by summed WEIGHT, never by row count -- the bank retains rejected events as dead rows
    carrying exactly w0 = 0 (0.9% of the T2K bank, some of which even pass the selection mask), so a
    count-based normalisation would silently include them and shift the scale.
    """
    spec = spec or SmearSpec()
    # SOFT TRUTH MEMBERSHIP IS RECTANGULAR-ONLY.  It spreads an event over cells using one shared edge
    # array per axis, which a staircase grid does not have -- its p_mu edges differ per cos slice.  Fail
    # loudly rather than index the wrong array and return a response matrix that looks fine.
    from adonis.unfold.binning import StaircaseGrid
    if soft_sigma is not None and isinstance(TG or truth_grid(), StaircaseGrid):
        raise NotImplementedError("soft_sigma is not supported on a StaircaseGrid truth binning")
    TG = TG or truth_grid()
    RG = RG or reco_grid()
    files = sorted(glob.glob(f"{bank_dir}/chunk_*.npz"))
    if max_chunks:
        files = files[:max_chunks]
    nch = BP.bank_nchunks(bank_dir)

    NF = n_flux()
    # RESPONSE carries a FLUX axis: a flux parameter scales signal as well as background, so the signal
    # term is bilinear, mu_i = sum_jb A_ijb c_j f_b, and A cannot be collapsed over b.
    A = np.zeros((RG.n, TG.n, NF))      # response: reco bin x true bin x flux bin
    n_true = np.zeros(TG.n)             # ALL true signal per truth bin (reco-selected or not) -> efficiency
    n_sig_reco = np.zeros(RG.n)         # signal that passes reco, per reco bin      -> purity numerator
    w_total = 0.0                       # total pre-selection rate (the 50k normalisation target)

    # BACKGROUND is kept as a COMPACT BANK, not as (chunk, index) pointers.  Its weight has to be
    # recomputed from the hard-vertex records at every fit iteration, so the records must stay resident;
    # pointers would mean re-reading the bank per iteration.  filter_events guarantees
    # bank_weight(filter_events(B, m), theta) == bank_weight(B, theta)[m], so compacting changes nothing.
    bkg_parts, bkg_bins, bkg_fbins, bkg_frac = [], [], [], []
    for ci, f in enumerate(files):
        B = BP.load_bank_chunk(f, nch)
        R = smear_chunk(B, spec, ci)
        w0 = np.asarray(B["w0"], dtype=float)
        w_total += w0.sum()

        tsel, tobs, _, _ = SG.select_full(B, signal)
        tdpt, tdat = _obs2(tobs)
        tbin = TG.index(tdpt, tdat)
        # RECO REPLICAS: replica r of chunk ci uses seed offset r, so each is an independent draw of the
        # same detector and the set is still a pure function of (seed, chunk).
        nrep = max(int(soft_reco), 1)
        reps = []
        for rep in range(nrep):
            Rr = smear_chunk(B, spec, ci * 1000 + rep) if soft_reco else R
            rsel_r, robs_r, _, _ = SG.select_full(Rr, signal)
            rdpt_r, rdat_r = _obs2(robs_r)
            reps.append((rsel_r, RG.index(rdpt_r, rdat_r)))
            del Rr
        rsel, rbin = reps[0]
        fbin = flux_index(np.asarray(B["k_nu"], dtype=float)[:, 0])   # TRUE E_nu, never the reco proxy

        # TRUE signal = passes the true selection AND lands inside the truth grid.  True signal outside
        # the grid has no template to scale it, so it is background by construction; calling it signal
        # would attach it to whichever c_j the clipping happened to choose.
        is_sig = tsel & (tbin >= 0)
        if soft_sigma is None:
            np.add.at(n_true, tbin[is_sig], w0[is_sig])
        else:
            mi = _soft_membership(tdpt[is_sig], TG.dpt, soft_sigma[0])
            mj = _soft_membership(tdat[is_sig], TG.dat, soft_sigma[1])
            n_true += ((mi[:, :, None] * mj[:, None, :]).reshape(int(is_sig.sum()), TG.n)
                       * w0[is_sig][:, None]).sum(axis=0)

        if soft_reco:
            # SIGNAL: every replica contributes 1/nrep of the weight to the bin it landed in.
            for rsel_r, rbin_r in reps:
                keep_r = rsel_r & (rbin_r >= 0)
                sr = keep_r & is_sig
                np.add.at(A, (rbin_r[sr], tbin[sr], fbin[sr]), w0[sr] / nrep)
                np.add.at(n_sig_reco, rbin_r[sr], w0[sr] / nrep)
            # BACKGROUND: the detector does not know which events are signal, so the background must be
            # softened the same way.  It cannot be replicated as EVENTS -- its weight is theta-dependent
            # and has to stay one row per event in the compact bank -- so instead each event keeps one
            # row and carries a DISTRIBUTION over reco bins.  The hard case is the one-hot special case
            # of this, which is why both paths end in the same matrix product downstream.
            any_pass = np.zeros(len(w0), bool)
            for rsel_r, rbin_r in reps:
                any_pass |= (rsel_r & (rbin_r >= 0))
            bkg_r = any_pass & ~is_sig          # non-signal that survives reco in AT LEAST one replica
            if bkg_r.any():
                bkg_parts.append(BP.filter_events(B, bkg_r))
                loc = np.cumsum(bkg_r) - 1      # global event index -> row in this chunk's compact bank
                rows, cols, vals = [], [], []
                for rsel_r, rbin_r in reps:
                    m = bkg_r & rsel_r & (rbin_r >= 0)
                    rows.append(loc[m]); cols.append(rbin_r[m] * NF + fbin[m])
                    vals.append(np.full(int(m.sum()), 1.0 / nrep))
                bkg_frac.append((np.concatenate(rows), np.concatenate(cols), np.concatenate(vals),
                                 int(bkg_r.sum())))
            if log:
                log(f"  [unfold] chunk {ci + 1}/{len(files)} ({nrep} replicas)")
            del B
            continue
        keep = rsel & (rbin >= 0)
        sig_r = keep & is_sig
        if soft_sigma is None:
            np.add.at(A, (rbin[sig_r], tbin[sig_r], fbin[sig_r]), w0[sig_r])
        else:
            # SOFT: one event lands on several truth cells, weighted by the resolution kernel.
            mi = _soft_membership(tdpt[sig_r], TG.dpt, soft_sigma[0])
            mj = _soft_membership(tdat[sig_r], TG.dat, soft_sigma[1])
            cell = (mi[:, :, None] * mj[:, None, :]).reshape(int(sig_r.sum()), TG.n)
            wcell = cell * w0[sig_r][:, None]
            for jj in range(TG.n):
                np.add.at(A[:, jj, :], (rbin[sig_r], fbin[sig_r]), wcell[:, jj])
        np.add.at(n_sig_reco, rbin[sig_r], w0[sig_r])

        bkg_r = keep & ~is_sig
        if bkg_r.any():
            bkg_parts.append(BP.filter_events(B, bkg_r))
            bkg_bins.append(rbin[bkg_r].astype(np.int32))
            bkg_fbins.append(fbin[bkg_r].astype(np.int32))
        if log:
            log(f"  [unfold] chunk {ci + 1}/{len(files)}  reco-sel {int(keep.sum()):6d}"
                f"  sig {int(sig_r.sum()):6d}  bkg {int(bkg_r.sum()):6d}")
        del B, R

    bkg_bank = SG._concat_compact(bkg_parts)
    if soft_reco:
        # one sparse (n_events x n_reco*n_flux) matrix: row e, column i*NF+b holds the fraction of
        # event e's weight the detector puts in reco bin i.  Rows may sum to LESS than 1 -- an event
        # that survives the selection in only some replicas is partly inefficient, and that is real.
        from scipy.sparse import coo_matrix
        R_, C_, V_, off = [], [], [], 0
        for r_, c_, v_, n_ in bkg_frac:
            R_.append(r_ + off); C_.append(c_); V_.append(v_); off += n_
        bkg_M = coo_matrix((np.concatenate(V_), (np.concatenate(R_), np.concatenate(C_))),
                           shape=(off, RG.n * NF)).tocsr()
        bkg_bin = np.zeros(0, np.int32); bkg_fbin = np.zeros(0, np.int32)
        n_bkg_reco = np.asarray(bkg_M.T @ np.asarray(bkg_bank["w0"], float)).reshape(RG.n, NF).sum(1)
    else:
        bkg_M = None
        bkg_bin = np.concatenate(bkg_bins)
        bkg_fbin = np.concatenate(bkg_fbins)
        n_bkg_reco = np.bincount(bkg_bin, weights=np.asarray(bkg_bank["w0"], float), minlength=RG.n)

    # ONE scale, applied to w0.  bank_weight is w0 * (knob factors), so scaling w0 carries the
    # normalisation into every theta-dependent background weight with no further bookkeeping.
    scale = 1.0 if norm_events is None else float(norm_events) / w_total
    A *= scale; n_true *= scale; n_sig_reco *= scale; n_bkg_reco *= scale
    bkg_bank["w0"] = np.asarray(bkg_bank["w0"], float) * scale

    with np.errstate(invalid="ignore", divide="ignore"):
        eff = np.where(n_true > 0, A.sum(axis=(0, 2)) / n_true, np.nan)
        tot = n_sig_reco + n_bkg_reco
        pur = np.where(tot > 0, n_sig_reco / tot, np.nan)
    return dict(A=A, n_true=n_true, n_sig_reco=n_sig_reco, n_bkg_reco=n_bkg_reco,
                bkg_bank=bkg_bank, bkg_bin=bkg_bin, bkg_fbin=bkg_fbin, bkg_M=bkg_M, nflux=NF,
                eff=eff, purity=pur, scale=scale, w_total_raw=w_total,
                norm_events=(w_total * scale))
