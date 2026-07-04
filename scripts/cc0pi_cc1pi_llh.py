"""CC0pi + CC1pi COMBINED exact LLH surface — does adding the CC1pi+Np STV channel break a CC0pi degeneracy?

Combined absolute chi^2 over FIVE T2K STV datasets, all re-summed from the ONE differentiable bank:
  CC0pi-Np : dpt, dat        (signal_cc0pi, per-nucleon 1e-38 units)
  CC1pi+Np : pN, dpTT, daT   (signal_cc1pi_stv, nb/unit per CH + FROZEN nominal free-H offset)

Dataset assembly is REUSED from analysis.t2k.differentiability.info_content.build_datasets (single source of
truth: same acceptance/units/free-H as the info-content joint fit).  To keep the fast reweight, the bank is
sliced to the UNION of the CC0pi and CC1pi signal events; each dataset indexes into that union slice.

The chi2 object exposes the SAME interface as cc0pi_llh_lib.make_chi2, so the 2D mapping + Hessian-ellipse +
non-Gaussianity machinery (cc0pi_llh_surface.compute_pair / make_figure) is reused unchanged.

  python -u scripts/cc0pi_cc1pi_llh.py [pairkey ...] [--ng 51]
  python -u scripts/cc0pi_cc1pi_llh.py --plot-only /tmp/adonis_llh/<key>_cc0cc1.npz
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))                       # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                   # repo root
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import cc0pi_llh_lib as L
import cc0pi_llh_surface as S
from analysis.t2k.differentiability import bank_plot as BP, bank_reweight as BR
from analysis.t2k.differentiability import info_content as IC
from analysis.t2k.differentiability.full_knobs import nominal_knobs

BANKDIR = os.environ.get("ADONIS_EVENT_BANK", "output/event_bank")


def make_chi2_combined(verbose=True):
    """Build a make_chi2-compatible dict for the 5-dataset CC0pi+CC1pi combined absolute chi^2, using a
    union-sliced bank for speed.  offsets (frozen free-H for CC1pi) are added per bin."""
    B = BP.load_bank(BANKDIR)
    ds = IC.build_datasets(B)                                                  # 5 datasets (sel_idx into full B)
    N = len(B["w0"])
    allsel = np.unique(np.concatenate([np.asarray(d["sel_idx"]) for d in ds])).astype(np.int64)
    pos = -np.ones(N, np.int64); pos[allsel] = np.arange(len(allsel))          # full -> union-slice position
    JB = BR.to_jax(B)
    JBu = {k: v[jnp.asarray(allsel)] for k, v in JB.items()}
    grids = BR.default_grids()
    if verbose:
        print(f"[comb] bank {N} ev -> union slice {len(allsel)} "
              f"(CC0pi {int(BP.signal_cc0pi(B)[0].sum())} + CC1pi {int(BP.signal_cc1pi_stv(B)[0].sum())})",
              flush=True)

    # per-dataset device tensors (indices into the union slice)
    STR = []
    datas = []; cinvs = []; conds = []
    for d in ds:
        sel_u = pos[np.asarray(d["sel_idx"])]
        assert (sel_u >= 0).all()
        STR.append(dict(sel=jnp.asarray(sel_u), bidx=jnp.asarray(np.asarray(d["binidx"])), nb=int(d["nbin"]),
                        scale=jnp.asarray(np.asarray(d["scale_bin"])), off=jnp.asarray(np.asarray(d["offset"])),
                        Ci=jnp.asarray(d["Cinv"])))
        datas.append(np.asarray(d["data"]))
        cinvs.append(np.asarray(d["Cinv"]))
        conds.append(float(np.linalg.cond(d["Cinv"])))
    data_all = jnp.asarray(np.concatenate(datas))
    import scipy.linalg as sla
    cinv_all = jnp.asarray(sla.block_diag(*cinvs))
    ndf = int(len(data_all))
    names = [d["name"] for d in ds]

    def model_vec(knobs):
        w = BR.weight_jit(JBu, knobs, grids)                                   # JBu operand -> finite
        outs = []
        for s in STR:
            mv = jax.ops.segment_sum(w[s["sel"]], s["bidx"], num_segments=s["nb"]) * s["scale"] + s["off"]
            outs.append(mv)
        return jnp.concatenate(outs)

    def chi2_abs(knobs):
        r = model_vec(knobs) - data_all
        return r @ cinv_all @ r

    def chi2_both(knobs):
        m = model_vec(knobs); ra = m - data_all
        A = (m @ cinv_all @ data_all) / (m @ cinv_all @ m)                     # single global A (units mixed:
        rs = A * m - data_all                                                  # secondary only, documented)
        return ra @ cinv_all @ ra, rs @ cinv_all @ rs, A

    if verbose:
        NOM = nominal_knobs()
        c0 = float(chi2_abs(NOM))
        print(f"[comb] nominal combined chi2/ndf = {c0/ndf:.3f} (ndf={ndf})", flush=True)
        for nm, cd in zip(names, conds):
            print(f"[comb] cov[{nm}] cond={cd:.2e}", flush=True)
    return dict(model_vec=model_vec, chi2_abs=chi2_abs, chi2_both=chi2_both, ndf=ndf,
                obs_names=names, conds=conds, nfloored=[0] * len(ds))


# CC1pi tightens RES -> zoom the combined norm-pair grid onto the now-narrow constraint (the CC0pi-only range
# res_norm (0.30,2.80) is far too wide once CC1pi is added).  maqe_axial is QE-only (CC1pi is orthogonal) so
# its CC0pi range is kept.
COMB_RANGES = {
    "qe_res_norm": ((0.76, 1.02), (0.45, 1.45)),
}


def main():
    ng = 51
    if "--ng" in sys.argv: ng = int(sys.argv[sys.argv.index("--ng") + 1])
    keys = [a for a in sys.argv[1:] if a in S.PAIRS]
    if not keys: keys = ["qe_res_norm", "maqe_axial"]                          # norm pair is the headline
    for k, (r0, r1) in COMB_RANGES.items():                                    # zoom combined grids
        n0, n1, t0_, t1_, _, _ = S.PAIRS[k]; S.PAIRS[k] = (n0, n1, t0_, t1_, r0, r1)
    t0 = time.time()
    print(f"[comb] pairs={keys} ng={ng}", flush=True)
    C = make_chi2_combined()
    for key in keys:
        npz = S.compute_pair(C, key, ng, "cc0cc1")
        S.make_figure(npz)
    print(f"[comb] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        S.make_figure(sys.argv[sys.argv.index("--plot-only") + 1])
    else:
        main()
