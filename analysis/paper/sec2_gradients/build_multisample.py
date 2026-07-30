"""Build ONE multi-sample per-bin gradient npz on carbon, in the physfit schema sec2_gradients/make.py
renders.  Samples (all on 12C): T2K nu CC0pi/CC1pi (the persisted Gate-I Jacobian) + pi+ -> C and
proton -> C beam-scattering cross sections (beam_fisher.beam_jacobian).

Every J shares the SAME 28-knob SPEC (physical_fit.SPEC): the beam J's come from jvp through
pool_fsi_reweight, so only the FSI knobs are non-zero and the hard-vertex/SF columns are EXACTLY zero
(no manual zero-fill).  Bins are concatenated along the observable axis; the combined Gate-I `shrink` is
recomputed from the STACKED Fisher F = (J/sigma)^T (J/sigma) + prior^-2 -- NOT carried per sample, so
make.py's red "passes Gate I" labels reflect what ALL these samples jointly constrain.

    python -m analysis.paper.sec2_gradients.build_multisample        # writes output/altgen/multisample_carbon.npz
    python -m analysis.paper.sec2_gradients.make multisample_carbon  # renders it
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "altgen"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # analysis/paper: beam_fisher bare-imports physical_fit
from analysis.paper import physical_fit as PF        # noqa: E402
from analysis.paper import style                      # noqa: E402
from analysis.paper.beams import beam_fisher as BF     # noqa: E402

# one dskeys entry per 15-bin block beam_jacobian returns (reaction bins, then the second observable)
BEAM_OBS = {"pip": ["pip_react", "pip_abs"], "prot": ["prot_react", "prot_pipro"]}

# persisted physfit-format npz samples (each already a bank -> jvp Jacobian), stacked in this order.
# Only the ones present on disk are used, so electron (physfit_electron) joins automatically once made.
NPZ_SAMPLES = ("physfit_gate1", "physfit_minerva", "physfit_electron")


def build(beams=("pip", "prot"), npz_samples=NPZ_SAMPLES, nbins=15, syst=0.05, out_label="multisample_carbon"):
    J, sigma, dskeys, row0 = [], [], [], [0]
    edges = {}                                           # {dskey}_edges -> per-obs bin edges (for --data marks)

    for lab in npz_samples:                              # bank-Jacobian samples (T2K, MINERvA, electron)
        f = style.ALTGEN / f"{lab}.npz"
        if not f.exists():
            print(f"  [skip] {lab}: no npz at {f}"); continue
        d = np.load(f, allow_pickle=True)
        assert [str(x) for x in d["pnames"]] == PF.PNAMES, f"{lab} knob order != physical_fit SPEC"
        J.append(np.asarray(d["J"])); sigma.append(np.asarray(d["sigma"]))
        r = np.asarray(d["row0"])
        for j, key in enumerate([str(x) for x in d["dskeys"]]):
            row0.append(row0[-1] + int(r[j + 1] - r[j])); dskeys.append(key)
            if f"{key}_edges" in d.files:
                edges[f"{key}_edges"] = np.asarray(d[f"{key}_edges"])

    for beam in beams:                                   # FSI-only beam Jacobians (pi+, proton)
        Jb, sb, _c, e_beam, _n = BF.beam_jacobian(beam, nbins=nbins, syst=syst)   # (2*nbins, 28)
        J.append(np.asarray(Jb)); sigma.append(np.asarray(sb))
        for key in BEAM_OBS[beam]:                       # two observables share the beam-momentum edges
            row0.append(row0[-1] + nbins); dskeys.append(key)
            edges[f"{key}_edges"] = np.asarray(e_beam)

    J = np.vstack(J); sigma = np.concatenate(sigma); prior = PF.PRIOR
    Jw = J / sigma[:, None]                             # error-weighted rows -> Fisher integrand
    F = Jw.T @ Jw
    V = np.linalg.inv(F + np.diag(1.0 / prior ** 2))   # marginalized posterior covariance
    shrink = np.sqrt(np.diag(V)) / prior               # Gate I: FIT when < 0.5 (on the COMBINED set)

    out = style.ALTGEN / f"{out_label}.npz"
    np.savez(out, J=J, sigma=sigma, prior=prior, pnames=PF.PNAMES,
             dskeys=dskeys, row0=np.asarray(row0), shrink=shrink, F=F, V=V, **edges)
    print(f"[out] {out}")
    print(f"  bins={J.shape[0]}  knobs={J.shape[1]}  observables={len(dskeys)}  ({'+'.join(('T2K',)+beams)})")
    print(f"  {int((shrink < 0.5).sum())}/{len(prior)} knobs pass Gate I on the combined set")
    return out


if __name__ == "__main__":
    build()
