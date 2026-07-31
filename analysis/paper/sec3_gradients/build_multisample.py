"""Build ONE multi-sample per-bin gradient npz on carbon, in the physfit schema sec3_gradients/make.py
renders.  Samples (all on 12C): T2K nu CC0pi/CC1pi (the persisted Gate-I Jacobian) + pi+ -> C and
proton -> C beam-scattering cross sections (beam_fisher.beam_jacobian).

Every J shares the SAME 28-knob SPEC (physical_fit.SPEC): the beam J's come from jvp through
pool_fsi_reweight, so only the FSI knobs are non-zero and the hard-vertex/SF columns are EXACTLY zero
(no manual zero-fill).  Bins are concatenated along the observable axis; the combined Gate-I `shrink` is
recomputed from the STACKED Fisher F = (J/sigma)^T (J/sigma) + prior^-2 -- NOT carried per sample, so
make.py's red "passes Gate I" labels reflect what ALL these samples jointly constrain.

    python -m analysis.paper.sec3_gradients.build_multisample        # writes output/altgen/multisample_carbon.npz
    python -m analysis.paper.sec3_gradients.make multisample_carbon  # renders it
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "altgen"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # analysis/paper: beam_fisher bare-imports physical_fit
from analysis.paper import physical_fit as PF        # noqa: E402
from analysis.paper import style                      # noqa: E402
from analysis.paper import fisher_engine as FE         # noqa: E402
from analysis.paper.beams import beam_fisher as BF     # noqa: E402

# one dskeys entry per 15-bin block beam_jacobian returns (reaction bins, then the second observable)
BEAM_OBS = {"pip": ["pip_react", "pip_abs"], "prot": ["prot_react", "prot_pipro"],
            "neut": ["neut_react", "neut_pipro"]}

# persisted physfit-format npz samples (each already a bank -> jvp Jacobian), stacked in this order.
# Each entry is (label, keep) where keep is None (all observables) or a tuple of dskeys to keep -- so a
# sample can be restricted to the variables the experiment ACTUALLY measured.  Only labels present on
# disk are used.  Every column is one real measurement with its own NUISANCE signal definition:
#   physfit_gate1        T2K CC0pi-Np + CC1pi+Np STV  -> keep the STV vars + the CC0pi muon 1D
#                        (pmu,cosmu); drop the CC1pi pion marginals (ppi,cospi) and multiplicities
#                        (n_p,n_chpi).  Both sec2 and sec3 use this 7-observable T2K set.
#   physfit_t2k_pcos     T2K CC0pi 2D d2sigma/dpmu dcosmu (hadron-inclusive, isT2K_CC0pi Analysis I)
#   physfit_minerva      MINERvA CC0pi-Np STV (isCC0piNp_MINERvA_STV)
#   physfit_minerva_ptpz MINERvA qelike muon pT/p|| (isCC0pi_MINERvAPTPZ, hadron-inclusive)
#   physfit_electron     (e,e') EM QE/RES omega
NPZ_SAMPLES = (
    ("physfit_gate1",        ("dpt", "dat", "pmu", "cosmu", "pn", "dptt", "daT")),  # T2K STV + CC0pi muon 1D (pmu,cosmu)
    # ("physfit_t2k_pcos",   None),                                  # T2K CC0pi 2D muon -- DISABLED (2D binning); driver + npz kept
    ("physfit_minerva",      None),                                   # MINERvA CC0pi-Np STV
    ("physfit_minerva_ptpz", ("mnv_ptmu", "mnv_pzmu")),              # MINERvA qelike: keep 1D pT/p|| (2D ptpl DISABLED)
    ("physfit_electron",     None),                                   # (e,e')
)

# NOTE: this ONE npz feeds both sec2 (the gradient figure) and sec3 (the Fisher subsets) -- both now use
# the SAME sample composition, T2K keep-filtered to its 7 STV+muon observables.  (A `multisample_carbon_full`
# variant with the full 11-obs T2K used to exist for sec3; it was retired when sec3 was aligned to sec2.)

def build(beams=("pip", "prot", "neut"), npz_samples=NPZ_SAMPLES, nbins=15, syst=0.05, out_label="multisample_carbon"):
    J, sigma, dskeys, row0 = [], [], [], [0]
    edges = {}                                           # {dskey}_edges -> per-obs bin edges (for --data marks)

    for lab, keep in npz_samples:                        # bank-Jacobian samples (T2K, MINERvA, electron)
        f = style.ALTGEN / f"{lab}.npz"
        if not f.exists():
            print(f"  [skip] {lab}: no npz at {f}"); continue
        d = np.load(f, allow_pickle=True)
        assert [str(x) for x in d["pnames"]] == PF.PNAMES, f"{lab} knob order != physical_fit SPEC"
        Jd = np.asarray(d["J"]); sd = np.asarray(d["sigma"]); r = np.asarray(d["row0"])
        for j, key in enumerate([str(x) for x in d["dskeys"]]):
            if keep is not None and key not in keep:
                continue                                 # drop observables the experiment did not report
            a, b = int(r[j]), int(r[j + 1])
            J.append(Jd[a:b]); sigma.append(sd[a:b])      # slice per observable so a subset is exact
            row0.append(row0[-1] + (b - a)); dskeys.append(key)
            if f"{key}_edges" in d.files:
                edges[f"{key}_edges"] = np.asarray(d[f"{key}_edges"])

    for beam in beams:                                   # FSI-only beam Jacobians (pi+, proton)
        Jb, sb, _c, e_beam, _n = BF.beam_jacobian(beam, nbins=nbins, syst=syst)   # (2*nbins, 28)
        J.append(np.asarray(Jb)); sigma.append(np.asarray(sb))
        for key in BEAM_OBS[beam]:                       # two observables share the beam-momentum edges
            row0.append(row0[-1] + nbins); dskeys.append(key)
            edges[f"{key}_edges"] = np.asarray(e_beam)

    J = np.vstack(J); sigma = np.concatenate(sigma); prior = PF.PRIOR
    F, V, _sig_post, shrink, _reach = FE.gate1(J, sigma, prior)   # Fisher + Gate I on the COMBINED set

    out = style.ALTGEN / f"{out_label}.npz"
    np.savez(out, J=J, sigma=sigma, prior=prior, pnames=PF.PNAMES,
             dskeys=dskeys, row0=np.asarray(row0), shrink=shrink, F=F, V=V, **edges)
    print(f"[out] {out}")
    print(f"  bins={J.shape[0]}  knobs={J.shape[1]}  observables={len(dskeys)}  ({'+'.join(('T2K',)+beams)})")
    print(f"  {int((shrink < 0.5).sum())}/{len(prior)} knobs pass Gate I on the combined set")
    return out


if __name__ == "__main__":
    build()                                                            # the shared sec2 + sec3 npz
