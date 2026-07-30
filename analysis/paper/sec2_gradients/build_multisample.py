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


def build(beams=("pip", "prot"), nbins=15, syst=0.05, out_label="multisample_carbon"):
    t = np.load(style.ALTGEN / "physfit_gate1.npz", allow_pickle=True)
    assert [str(x) for x in t["pnames"]] == PF.PNAMES, "T2K npz knob order != physical_fit SPEC"
    J = [np.asarray(t["J"])]                            # (nbin_T2K, 28)
    sigma = [np.asarray(t["sigma"])]
    dskeys = [str(x) for x in t["dskeys"]]
    row0 = list(np.asarray(t["row0"]))                 # cumulative bin boundaries, ends at nbin_T2K

    for beam in beams:
        Jb, sb, _c, _e, _n = BF.beam_jacobian(beam, nbins=nbins, syst=syst)   # (2*nbins, 28)
        J.append(np.asarray(Jb)); sigma.append(np.asarray(sb))
        for key in BEAM_OBS[beam]:                       # two observables, nbins each
            row0.append(row0[-1] + nbins); dskeys.append(key)

    J = np.vstack(J); sigma = np.concatenate(sigma); prior = PF.PRIOR
    Jw = J / sigma[:, None]                             # error-weighted rows -> Fisher integrand
    F = Jw.T @ Jw
    V = np.linalg.inv(F + np.diag(1.0 / prior ** 2))   # marginalized posterior covariance
    shrink = np.sqrt(np.diag(V)) / prior               # Gate I: FIT when < 0.5 (on the COMBINED set)

    out = style.ALTGEN / f"{out_label}.npz"
    np.savez(out, J=J, sigma=sigma, prior=prior, pnames=PF.PNAMES,
             dskeys=dskeys, row0=np.asarray(row0), shrink=shrink, F=F, V=V)
    print(f"[out] {out}")
    print(f"  bins={J.shape[0]}  knobs={J.shape[1]}  observables={len(dskeys)}  ({'+'.join(('T2K',)+beams)})")
    print(f"  {int((shrink < 0.5).sum())}/{len(prior)} knobs pass Gate I on the combined set")
    return out


if __name__ == "__main__":
    build()
