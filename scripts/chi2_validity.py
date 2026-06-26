"""Re-evaluate the validity of the ADoNIS-vs-ACHILLES chi2: with spiky importance weights a bin can
have few EFFECTIVE entries (Kish N_eff=(Sum w)^2/Sum w^2) even with many raw events, so the analytic
Gaussian error sqrt(Sum w^2) -- and hence the per-bin chi2 -- may be unreliable.  This script, per
observable/bin and for BOTH figures:

  1. N_eff (ADoNIS spiky side; ACHILLES ~unweighted) -> flag bins with N_eff < N_MIN (Gaussian rule).
  2. BOOTSTRAP test of the Gaussian error: resample the ADoNIS events B times; compare the empirical
     bin-content std (and skew) to the analytic sqrt(Sum w^2).  ratio~1 & |skew| small => Gaussian valid.
  3. chi2/ndf recomputed (a) with analytic errors, (b) with bootstrap errors, (c) on reliable bins only.
     If (a)~(b) and dropping unreliable bins doesn't move it, the quoted chi2 is trustworthy.

Usage: python -u scripts/chi2_validity.py [B=600]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

B = int(sys.argv[1]) if len(sys.argv) > 1 else 600
N_MIN = 25                  # Gaussian-validity threshold on effective entries per bin
RNG = np.random.default_rng(0)


def _hist(v, w, edg):
    sw, _ = np.histogram(v, edg, weights=w)
    sw2, _ = np.histogram(v, edg, weights=w ** 2)
    return sw, sw2


def boot_err(v, w, edg, B):
    """Bootstrap the weighted bin content: empirical std + skew of Sum w over B resamples of the events."""
    n = len(w); nb = len(edg) - 1
    idx = np.digitize(v, edg) - 1
    samples = np.zeros((B, nb))
    for b in range(B):
        pick = RNG.integers(0, n, n)                       # resample events with replacement
        ib = idx[pick]; wb = w[pick]
        m = (ib >= 0) & (ib < nb)
        samples[b] = np.bincount(ib[m], wb[m], minlength=nb)
    mu = samples.mean(0); sd = samples.std(0)
    sk = np.where(sd > 0, ((samples - mu) ** 3).mean(0) / np.maximum(sd ** 3, 1e-300), 0.0)
    return sd, sk


def report(name, A, H, VARS):
    print(f"\n================  {name}  ================")
    print(f"{'var':8s} {'minNeff':>7} {'bins<25':>7} | {'<bootSD/anaSD>':>14} {'maxSkew':>7} | "
          f"{'chi2_ana':>8} {'chi2_boot':>9} {'chi2_Neff>=25':>13}")
    for k, edg, _ in VARS:
        bw = np.diff(edg)
        swA, sw2A = _hist(A[k], A["w"], edg); neff = np.where(sw2A > 0, swA ** 2 / np.maximum(sw2A, 1e-300), 0.0)
        swH, sw2H = _hist(H[k], H["w"], edg)
        da, ea = swH / bw, np.sqrt(sw2H) / bw
        dd, ea_ana = swA / bw, np.sqrt(sw2A) / bw            # ADoNIS analytic (Gaussian) error
        sd_boot, skew = boot_err(A[k], A["w"], edg, B)
        ed_boot = sd_boot / bw
        m = (da > 0) & (dd > 0)
        chi2_ana = np.sum((da[m] - dd[m]) ** 2 / (ea[m] ** 2 + ea_ana[m] ** 2))
        chi2_boot = np.sum((da[m] - dd[m]) ** 2 / (ea[m] ** 2 + ed_boot[m] ** 2))
        rel = m & (neff >= N_MIN)
        chi2_rel = np.sum((da[rel] - dd[rel]) ** 2 / (ea[rel] ** 2 + ea_ana[rel] ** 2))
        ndf, ndr = int(m.sum()), int(rel.sum())
        ratio = np.mean((sd_boot[m] / np.maximum(np.sqrt(sw2A)[m], 1e-300)))   # boot/analytic err ratio
        print(f"{k:8s} {neff[neff>0].min():7.0f} {int((neff[m]<25).sum()):7d} | {ratio:14.2f} {np.abs(skew[m]).max():7.2f} | "
              f"{chi2_ana/max(ndf,1):8.2f} {chi2_boot/max(ndf,1):9.2f} {chi2_rel/max(ndr,1):8.2f}/{ndr:<4d}")


def main():
    # CC1pi
    import scripts.cc1pi_engine_plot as P, scripts.cc1pi_signal as S
    P.BANK = "data/oracle/t2k_cc1pi.npz"        # module read sys.argv at import; pin the bank
    A_res = P.engine_signal(P.BANK); A_qe = P.qe_created_signal()
    A1 = {k: np.concatenate([A_res[k], A_qe[k]]) for k in A_res}
    ach = dict(np.load("data/oracle/t2k_cc1pi_rich_ach_FSI.npz"))
    H1 = S.ach_select(ach, dict(S.DEFAULT)); H1 = {**H1, "w": H1["w"] * float(ach["weight_to_nb"])}
    report("CC1pi (RES + QE-created-pi)", A1, H1, P.VARS)
    # CC0pi
    import scripts.cc0pi_engine_combined as C
    qe = C._select_engine("data/oracle/t2k_cc0pi.npz")
    res = C._select_engine("data/oracle/t2k_cc1pi.npz")
    A0 = {k: np.concatenate([qe[k], res[k]]) for k in qe}; H0 = C._select_ach()
    report("CC0pi (QE + RES-absorbed)", A0, H0, C.VARS)
    print(f"\nLegend: chi2_ana = analytic-Gaussian (quoted); chi2_boot = with bootstrap errors; "
          f"chi2_Neff>=25 = reliable-bins-only (ndf).  bootSD/anaSD~1 & small skew & ana~boot => Gaussian VALID.")


if __name__ == "__main__":
    main()
