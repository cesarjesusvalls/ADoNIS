"""Mono 1 GeV free-proton: normalised-shape comparison ACHILLES hepmc vs ADoNIS forward,
for cos(theta_pi) [Npi-CM, rel q], cos(theta_mu) [lab, rel beam], W, Q2.
CHUNKED amps2 with a live progress line so status is always visible.
Usage: python scripts/angle_compare.py <N_adonis>   (use a tiny N first as a smoke test)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import scripts.test_3body_q2_measure as T
import scripts.mono_dist_compare as M
from adonis.xsec.dcc_current import exclusive_amps2_batch
from adonis.xsec.backend import flux_factor, MASS_PDG_PROTON


def adonis_forward(N, nchunks):
    chunk = max(N // nchunks, 1)
    cols = {k: [] for k in ("knu", "kmu", "pN", "ppi", "w")}
    t0 = time.time()
    for i in range(nchunks):
        rng = np.random.default_rng(i)
        knu = np.tile(T.k_nu, (chunk, 1)); pstk = np.tile(T.p_st, (chunk, 1))
        q2, j3, kmu, pN, ppi, val = T.isotropic_full(chunk, rng)
        a = np.zeros(chunk); idx = np.where(val)[0]
        if len(idx):
            a[idx] = np.asarray(exclusive_amps2_batch(knu[idx], kmu[idx], pstk[idx], pN[idx], ppi[idx], +1, 211))
        a = np.where(np.isfinite(a), a, 0)
        fl = np.asarray(flux_factor(knu, pstk, had_mass=MASS_PDG_PROTON))
        w = np.where(np.isfinite(a * fl * 0.5 * j3 * val), a * fl * 0.5 * j3 * val, 0)
        keep = w > 0
        cols["knu"].append(knu[keep]); cols["kmu"].append(kmu[keep])
        cols["pN"].append(pN[keep]); cols["ppi"].append(ppi[keep]); cols["w"].append(w[keep])
        print(f"  [adonis] {100*(i+1)/nchunks:5.1f}%  ({(i+1)*chunk:,}/{nchunks*chunk:,})  {time.time()-t0:.0f}s", flush=True)
    return {k: np.concatenate(v) for k, v in cols.items()}


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    nchunks = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print(f"=== angle_compare N={N} ===", flush=True)
    print("parsing ACHILLES mono hepmc ...", flush=True)
    aknu, akmu, apN, appi = M.parse_ach()
    aQ2, aW, acpi, acmu = M.variables(aknu, akmu, apN, appi)
    aw = np.ones(len(aknu))
    print(f"  ACHILLES events: {len(aknu)}", flush=True)
    print(f"generating ADoNIS forward ({nchunks} chunks) ...", flush=True)
    d = adonis_forward(N, nchunks)
    dQ2, dW, dcpi, dcmu = M.variables(d["knu"], d["kmu"], d["pN"], d["ppi"]); dw = d["w"]
    print(f"  ADoNIS forward events: {len(dw)}", flush=True)

    def cmp(name, av, dv, bins):
        ha, _ = np.histogram(av, bins=bins, weights=aw); ha = ha / ha.sum()
        hd, _ = np.histogram(dv, bins=bins, weights=dw); hd = hd / hd.sum()
        c = 0.5 * (bins[1:] + bins[:-1])
        print(f"\n{name}  (normalised; ratio=ADO/ACH):")
        for i in range(len(c)):
            if ha[i] > 0:
                print(f"  {c[i]:9.3f}   ACH {ha[i]:.4f}  ADO {hd[i]:.4f}   ratio {hd[i]/ha[i]:.3f}")
    cmp("cos_pi (Npi-CM rel q)", acpi, dcpi, np.linspace(-1, 1, 11))
    cmp("cos_mu (lab rel beam)", acmu, dcmu, np.linspace(0.85, 1.0, 16))
    cmp("Q2 [GeV^2]", aQ2, dQ2, np.linspace(0, 1.0, 11))
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
