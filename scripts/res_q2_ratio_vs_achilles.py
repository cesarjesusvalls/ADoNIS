"""High-stats dsigma/dQ2: real ACHILLES events (nofsi_res.hepmc) vs ADoNIS res_xsec.
Q2 = -(k_nu - k_mu)^2 with k_nu = the MAX-ENERGY pid14 (the status-2 beam, NOT the E=30
status-4 placeholder) and k_mu = pid13 status1.  ACHILLES events are unweighted, so each
carries sigma_total/N_events (sigma_total read from the run = 1.6947472705414825e-5 nb).
ADoNIS: res_xsec.generate(return_events=True), Q2 from the bare leptonic q, weight = w.
Prints the ratio ADoNIS/ACHILLES per Q2 bin with MC errors on both."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
from adonis.xsec import res_xsec as R

ACH_SIGMA = 1.6947472705414825e-5  # nb, read from run_nofsi_res Total xsec
HEPMC = Path(__file__).resolve().parents[2] / "Achilles/_resrun_out/nofsi_res.hepmc"


def parse_achilles_q2(path):
    """Yield Q2 [GeV^2] per event: k_nu = max-E pid14, k_mu = pid13 status1."""
    q2s = []
    nu_cands = []; mu = None
    def flush():
        nonlocal nu_cands, mu
        if nu_cands and mu is not None:
            knu = max(nu_cands, key=lambda p: p[0])
            q = knu - mu
            q2s.append((q[1]**2 + q[2]**2 + q[3]**2 - q[0]**2) / 1e6)  # -q^2 in GeV^2
        nu_cands = []; mu = None
    with open(path) as fh:
        for line in fh:
            t = line[:2]
            if t == "E ":
                flush()
            elif t == "P ":
                f = line.split()
                pid = int(f[3]); status = int(f[9])
                p4 = np.array([float(f[7]), float(f[4]), float(f[5]), float(f[6])])
                if pid == 14:
                    nu_cands.append(p4)
                elif pid == 13 and status == 1:
                    mu = p4
    flush()
    return np.array(q2s)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000000
    print(f"parsing ACHILLES events from {HEPMC.name} ...")
    q2_ach = parse_achilles_q2(HEPMC)
    nev = len(q2_ach)
    w_ach = np.full(nev, ACH_SIGMA / nev)
    print(f"  {nev} ACHILLES events, <Q2>={q2_ach.mean():.4f} GeV^2")

    print(f"generating ADoNIS res_xsec events (n={n}/channel) ...")
    r = R.generate(n, seed=0, return_events=True)
    ev = r["events"]
    q = ev["k_nu"] - ev["k_mu"]
    q2_ado = (np.sum(q[:, 1:]**2, axis=1) - q[:, 0]**2) / 1e6  # GeV^2
    w_ado = ev["w"]
    print(f"  ADoNIS total sigma = {r['sigma']:.4e}  (ratio {r['sigma']/ACH_SIGMA:.3f})")

    bins = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.9, 1.4, 2.5, 5.0])
    print("\n Q2 bin [GeV^2]    ACH dsig      ADO dsig     ratio ADO/ACH   (+/- )")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]; bw = hi - lo
        ma = (q2_ach >= lo) & (q2_ach < hi)
        md = (q2_ado >= lo) & (q2_ado < hi)
        sa = w_ach[ma].sum() / bw
        sd = w_ado[md].sum() / bw
        # errors
        ea = np.sqrt((w_ach[ma]**2).sum()) / bw
        ed = np.sqrt((w_ado[md]**2).sum()) / bw
        ratio = sd / sa if sa > 0 else 0
        rerr = ratio * np.sqrt((ed/sd)**2 + (ea/sa)**2) if sd > 0 and sa > 0 else 0
        print(f" [{lo:.2f},{hi:.2f})   {sa:.4e}   {sd:.4e}    {ratio:.3f}        {rerr:.3f}")
    # integrated low vs high Q2
    for cut in (0.2, 0.4):
        la = w_ach[q2_ach < cut].sum(); ld = w_ado[q2_ado < cut].sum()
        ha = w_ach[q2_ach >= cut].sum(); hd = w_ado[q2_ado >= cut].sum()
        print(f"\n Q2<{cut}: ADO/ACH = {ld/la:.3f}   Q2>={cut}: ADO/ACH = {hd/ha:.3f}")


if __name__ == "__main__":
    main()
