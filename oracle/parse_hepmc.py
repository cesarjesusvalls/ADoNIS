"""Parse the ACHILLES single-pion oracle events (NuHepMC Asciiv3 text) into
differential distributions: dsigma/dQ^2, dsigma/dW, and the pion-momentum and
pion-cos(theta) spectra.

These are the bin-by-bin ORACLE TARGETS used to validate the differentiable DCC
cross-section assembly (diffpi/dcc.py + the hadron-tensor port) at nominal knobs.

The event record (per NuHepMC):
  E <evt> <nvtx> <npart>
  W <weight>                      per-event weight (CV)
  A 0 GenCrossSection <xs> ...    running xsec estimate [pb] (HepMC convention)
  P <id> <vtx> <pid> px py pz E m <status>   (MEV)
Status: 4 = incoming beam lepton, 20 = target nucleus, 1 = final-state,
        2 = intermediate/struck.  PIDs: 11 = e-, 111 = pi0, 211/-211 = pi+/-,
        2112 = n, 2212 = p.

Convention check (HepMC3/NuHepMC): GenCrossSection carries the total cross
section in pb; the per-event weights sum to that total over the whole sample, so
a weighted histogram divided by nothing already integrates to sigma_tot.  We
instead normalise so the integral equals the reference total (122.8 nb) to be
robust to the weight bookkeeping, and report both.

Run:  python oracle/parse_hepmc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HEPMC = Path("/Users/cjesus/Software/DiffSinglePiProd/Achilles/res_1pi_12C.hepmc")
OUT = Path(__file__).resolve().parent / "oracle_distributions.npz"

PI_PIDS = {111, 211, -211}
LEPTON = 11  # e-


def parse_events(path: Path):
    """Yield dicts with weight, gen_xs, and the relevant 4-momenta per event."""
    evt = None
    with open(path) as fh:
        for line in fh:
            tag = line[:2]
            if tag == "E ":
                if evt is not None:
                    yield evt
                evt = {"w": None, "gen_xs": None, "parts": []}
            elif evt is None:
                continue
            elif tag == "W ":
                evt["w"] = float(line.split()[1])
            elif tag == "A " and "GenCrossSection" in line:
                evt["gen_xs"] = float(line.split()[3])
            elif tag == "P ":
                f = line.split()
                # P id vtx pid px py pz E m status
                pid = int(f[3])
                p4 = np.array([float(f[7]), float(f[4]), float(f[5]), float(f[6])])  # (E,px,py,pz)
                status = int(f[9])
                evt["parts"].append((pid, status, p4))
    if evt is not None:
        yield evt


def minkowski2(p):
    """p=(E,px,py,pz) -> p.p with (+,-,-,-)."""
    return p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2


def event_kinematics(evt):
    """Return (Q2 [MeV^2], W [MeV], p_pi [MeV], cos_theta_pi) or None."""
    parts = evt["parts"]
    k_in = k_out = p_pi = p_struck = None
    n_pi_fs = 0
    for pid, status, p4 in parts:
        if abs(pid) == LEPTON and status == 4:
            k_in = p4
        elif abs(pid) == LEPTON and status == 1:
            k_out = p4
        elif pid in PI_PIDS and status == 1:
            p_pi = p4
            n_pi_fs += 1
        elif abs(pid) in (2112, 2212) and status == 2:
            p_struck = p4
    if k_in is None or k_out is None or p_pi is None:
        return None
    q = k_in - k_out
    Q2 = -minkowski2(q)
    # W from the struck-nucleon + q system (resonance invariant mass).
    if p_struck is not None:
        W = np.sqrt(max(minkowski2(q + p_struck), 0.0))
    else:
        W = np.nan
    p_pi_mag = np.sqrt(p_pi[1] ** 2 + p_pi[2] ** 2 + p_pi[3] ** 2)
    # pion angle w.r.t. the momentum transfer q (3-vector)
    q3 = q[1:]
    cos_th = float(np.dot(p_pi[1:], q3) / (p_pi_mag * np.linalg.norm(q3) + 1e-12))
    return Q2, W, p_pi_mag, cos_th, n_pi_fs


def main():
    if not HEPMC.exists():
        sys.exit(f"hepmc not found: {HEPMC}")
    Q2s, Ws, ppis, costh, weights = [], [], [], [], []
    n_pi_counts = []
    last_gen_xs = None
    n_evt = 0
    for evt in parse_events(HEPMC):
        n_evt += 1
        if evt["gen_xs"] is not None:
            last_gen_xs = evt["gen_xs"]
        kin = event_kinematics(evt)
        if kin is None:
            continue
        Q2, W, p_pi, cth, n_pi = kin
        Q2s.append(Q2); Ws.append(W); ppis.append(p_pi)
        costh.append(cth); weights.append(evt["w"]); n_pi_counts.append(n_pi)
    Q2s = np.array(Q2s); Ws = np.array(Ws); ppis = np.array(ppis)
    costh = np.array(costh); weights = np.array(weights)
    n_pi_counts = np.array(n_pi_counts)

    print(f"events parsed:           {n_evt}")
    print(f"events with FS pion:     {len(Q2s)}")
    print(f"final-state pion mult:   {np.bincount(n_pi_counts)}")
    print(f"weight: min/mean/max     {weights.min():.4e} / {weights.mean():.4e} / {weights.max():.4e}")
    print(f"GenCrossSection (last):  {last_gen_xs:.6e} (HepMC pb units)")
    sum_w = weights.sum()
    print(f"sum of weights:          {sum_w:.6e}")
    print(f"Q2  [MeV^2] range:       {Q2s.min():.3e} .. {Q2s.max():.3e}")
    print(f"W   [MeV]   range:       {np.nanmin(Ws):.1f} .. {np.nanmax(Ws):.1f}")
    print(f"p_pi[MeV]   range:       {ppis.min():.1f} .. {ppis.max():.1f}")

    # Normalise so the histograms integrate to the reference total cross section.
    REF_TOTAL_NB = 122.82  # oracle/res_1pi_12C_reference.txt (sum of 4 channels)
    norm = REF_TOTAL_NB / sum_w  # weight -> nb such that sum -> REF_TOTAL_NB
    w_nb = weights * norm

    # Histogram definitions (oracle targets)
    Q2_edges = np.linspace(0.0, np.percentile(Q2s, 99.5), 31)        # MeV^2
    W_edges = np.linspace(1080.0, np.nanpercentile(Ws, 99.5), 31)    # MeV
    ppi_edges = np.linspace(0.0, np.percentile(ppis, 99.5), 31)      # MeV
    cth_edges = np.linspace(-1.0, 1.0, 21)

    def dsig(vals, edges, w):
        m = np.isfinite(vals)
        h, _ = np.histogram(vals[m], bins=edges, weights=w[m])
        width = np.diff(edges)
        return h / width  # dsigma/dx  [nb / unit]

    dsig_Q2 = dsig(Q2s, Q2_edges, w_nb)
    dsig_W = dsig(Ws, W_edges, w_nb)
    dsig_ppi = dsig(ppis, ppi_edges, w_nb)
    dsig_cth = dsig(costh, cth_edges, w_nb)

    np.savez(
        OUT,
        Q2_edges=Q2_edges, dsig_Q2=dsig_Q2,
        W_edges=W_edges, dsig_W=dsig_W,
        ppi_edges=ppi_edges, dsig_ppi=dsig_ppi,
        cth_edges=cth_edges, dsig_cth=dsig_cth,
        ref_total_nb=REF_TOTAL_NB, n_signal=len(Q2s),
    )
    print(f"\nintegral check (nb):     "
          f"Q2={np.sum(dsig_Q2*np.diff(Q2_edges)):.2f}  "
          f"W={np.nansum(dsig_W*np.diff(W_edges)):.2f}  "
          f"ppi={np.sum(dsig_ppi*np.diff(ppi_edges)):.2f}  "
          f"(ref {REF_TOTAL_NB})")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
