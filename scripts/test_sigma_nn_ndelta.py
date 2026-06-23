"""STANDALONE integration-accuracy test: ADoNIS sigma_nn_ndelta vs ACHILLES SigmaNN2NDelta.

No cascade, no events -- compares the NN->NDelta cross section sigma(sqrts) for Delta++ (isofactor 1,
pcm=1 GeV) directly:
  ACHILLES: SigmaNN2NDelta(sqrts, 1GeV, Delta++) via its adaptive DoubleExponential integrator, dumped
            over a fine sqrts grid by achilles:vertex with ACHILLES_SIGMADUMP=1 (the `SIGNND` lines,
            printed once at NucleonNucleon construction -- so we kill the container as soon as they land).
  ADoNIS  : nn_inelastic._CACHE["sig"] base table (fixed-grid trapezoid), interpolated to the same sqrts.
Isolates the JAX-forced fixed-grid-quadrature divergence (docs/logbook/jax_forced_divergences.md).

Run: python -u scripts/test_sigma_nn_ndelta.py            (runs ACHILLES + compares + plots)
     python -u scripts/test_sigma_nn_ndelta.py <ach.txt>  (reuse an existing SIGNND dump)
"""
import os, sys, re, time, subprocess
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

TAG = "C"; STEM = f"T2K_{TAG}_fate_gauss"
IMAGE = os.environ.get("ACH_VERTEX_IMAGE", "achilles:vertex")
DUMP = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ach_signnd.txt"


def run_achilles_dump():
    """Launch achilles:vertex with SIGMADUMP, collect the SIGNND lines (printed at cascade init), stop."""
    base = Path(f"_oracle_out/run_{STEM}.yml").read_text()
    card = re.sub(r"NEvents:\s*\d+", "NEvents: 1000", base, count=1)
    card = card.replace('Options: !include "data/default/OptionDefaults.yml"',
                        "Options:\n  Initialize:\n    Seed: 1\n    Accuracy: 1e-2\n"
                        "  Unweighting:\n    Name: Percentile\n    percentile: 99")
    Path(f"_oracle_out/run_{STEM}_sigdump.yml").write_text(card)
    log = "/tmp/ach_sigmadump_run.log"; cname = "achsigdump"
    subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
    print(f"[ach] running {IMAGE} (SIGMADUMP) -> {log}", flush=True)
    with open(log, "w") as fh:
        proc = subprocess.Popen(["docker", "run", "--name", cname, "-e", "ACHILLES_SIGMADUMP=1",
                                 "-v", f"{os.getcwd()}/_oracle_out:/out", "--entrypoint",
                                 "/achilles/bin/achilles", IMAGE, f"/out/run_{STEM}_sigdump.yml"],
                                stdout=fh, stderr=subprocess.STDOUT)
        for _ in range(150):                                       # poll for the SIGNND block (lands at init)
            time.sleep(2)
            n = sum(1 for l in open(log) if l.startswith("SIGNND"))
            if n >= 200:
                break
        subprocess.run(["docker", "kill", cname], capture_output=True)
        proc.wait(timeout=30)
    subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
    rows = [l for l in open(log) if l.startswith("SIGNND")]
    Path(DUMP).write_text("".join(rows))
    print(f"[ach] collected {len(rows)} SIGNND lines -> {DUMP}", flush=True)


def main():
    if not Path(DUMP).exists() or len(sys.argv) <= 1:
        run_achilles_dump()
    rs, ach = [], []
    for l in open(DUMP):
        m = re.match(r"SIGNND rs=(\S+) sig=(\S+)", l)
        if m:
            rs.append(float(m.group(1))); ach.append(float(m.group(2)))
    rs = np.array(rs); ach = np.array(ach)
    from adonis.fsi import nn_inelastic as nni
    nni._build_tables()
    ado = np.interp(rs, np.asarray(nni._CACHE["S"]), np.asarray(nni._CACHE["sig"]), left=0.0, right=0.0)
    thr = 2 * nni.MN_AVG + nni.MPI_AVG
    m = (ach > 0) & (ado > 0)                                      # both nonzero (above threshold)
    ratio = np.where(m, ado / np.where(ach == 0, np.nan, ach), np.nan)
    # report bands
    print(f"\nNN->NDelta sigma(sqrts) Delta++  threshold ~ {thr:.4f} GeV  ({m.sum()} valid points)")
    print(f"{'sqrts band':>16} {'ACH<sig>':>11} {'ADO<sig>':>11} {'ADO/ACH':>9} {'max|dev|%':>9}")
    bands = [(thr, 2.10), (2.10, 2.20), (2.20, 2.40), (2.40, 3.0), (3.0, 4.0)]
    for lo, hi in bands:
        b = m & (rs >= lo) & (rs < hi)
        if not b.any():
            continue
        dev = np.abs(ratio[b] - 1) * 100
        print(f"  [{lo:.3f},{hi:.2f}) {ach[b].mean():11.3e} {ado[b].mean():11.3e} "
              f"{np.nanmean(ratio[b]):9.4f} {dev.max():9.2f}")
    gm = m & (rs >= thr)
    print(f"\noverall (sqrts>thr): mean ADO/ACH = {np.nanmean(ratio[gm]):.4f}  "
          f"max|dev| = {(np.abs(ratio[gm]-1)*100).max():.2f}%  "
          f"RMS|dev| = {np.sqrt(np.nanmean((ratio[gm]-1)**2))*100:.2f}%")
    # plot
    fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    ax[0].plot(rs[m], ach[m], "-", label="ACHILLES (adaptive)", lw=1.8)
    ax[0].plot(rs[m], ado[m], "--", label="ADoNIS (fixed grid)", lw=1.8)
    ax[0].axvline(thr, color="grey", ls=":", label=f"threshold {thr:.3f}")
    ax[0].set_ylabel(r"$\sigma_{NN\to N\Delta^{++}}$ [mb] (pcm=1 GeV)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("NN->NDelta integration: ADoNIS sigma_nn_ndelta vs ACHILLES SigmaNN2NDelta")
    ax[1].plot(rs[m], ratio[m], "-", color="C3"); ax[1].axhline(1, color="k", lw=0.8)
    ax[1].axhline(1.01, color="grey", ls=":"); ax[1].axhline(0.99, color="grey", ls=":")
    ax[1].axvline(thr, color="grey", ls=":")
    ax[1].set_ylabel("ADoNIS / ACHILLES"); ax[1].set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax[1].set_ylim(0.9, 1.1); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("/tmp/sigma_nn_ndelta_test.png", dpi=120); plt.close(fig)
    print("\nwrote /tmp/sigma_nn_ndelta_test.png")


if __name__ == "__main__":
    main()
