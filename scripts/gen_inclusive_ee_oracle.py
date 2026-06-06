"""Parse the ACHILLES RES (e,e') inclusive oracle hepmc -> dsigma/domega CSV (Phase B2 oracle).

Runs (separately, in the achilles:oracle image) the RES_Spectral_Func electron-scattering
config `_oracle_out/inclusive_ee_12C_res.yml` (E=2.222 GeV, theta in [14,17] deg, a narrow
acceptance around 15.541 deg via the ACHILLES AngleTheta HardCut).  This parses the produced
hepmc into the energy-transfer spectrum omega = E_beam - E_e', weighted, which is the 1pi/Delta
half of the inclusive (e,e') Fig 1 -- the ACHILLES oracle for `onepi_dsigma_domega`.

  docker run --rm --platform linux/amd64 -w /achilles -v "$PWD/_oracle_out:/out" \
    --entrypoint /achilles/bin/achilles ghcr.io/cesarjesusvalls/achilles:oracle \
    /out/inclusive_ee_12C_res.yml
  python scripts/gen_inclusive_ee_oracle.py
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adonis.data.oracle.parse_hepmc import parse_events

E0 = 2222.0
HEPMC = ROOT / "_oracle_out" / "inclusive_ee_12C_res.hepmc"
CSV = ROOT / "data" / "oracle" / "inclusive_ee_12C_res.csv"


def omega_spectrum(hepmc=HEPMC, e0=E0, bins=22, w_lo=250.0, w_hi=900.0):
    """(omega_centres[MeV], dsigma/domega shape [peak-normalised], n_events)."""
    oms, ws = [], []
    for ev in parse_events(Path(hepmc)):
        els = [(p4, st) for pid, st, p4 in ev["parts"] if pid == 11]
        sc = [p4 for p4, st in els if st == 1] or [p4 for p4, st in els if p4[0] < e0 - 1]
        if not sc:
            continue
        oms.append(e0 - sc[0][0]); ws.append(ev["w"])
    oms, ws = np.array(oms), np.array(ws)
    h, edges = np.histogram(oms, bins=bins, range=(w_lo, w_hi), weights=ws)
    cen = 0.5 * (edges[1:] + edges[:-1])
    return cen, (h / h.max() if h.max() else h), len(oms)


if __name__ == "__main__":
    cen, shape, n = omega_spectrum()
    CSV.write_text(
        f"# ACHILLES RES_Spectral_Func inclusive (e,e') on 12C, E=2.222 GeV, theta in [14,17] deg\n"
        f"# (achilles:oracle image, {n} events). omega = E_beam - E_e'. Delta/1pi bump.\n"
        f"# Columns: omega[MeV], dsigma_domega_shape (peak-normalised)\n"
        + "".join(f"{c:.1f}  {s:.4f}\n" for c, s in zip(cen, shape)))
    print(f"wrote {CSV}  ({n} events); omega peak at {cen[int(np.argmax(shape))]:.0f} MeV")
