"""Build a T2K-like CC0pi-Np STV fake dataset from a GENIE gst tree.

GENUINE foreign-generator sample: GENIE 3.04 AR23_20i_00_000, numu on 12C, T2K-numu
flux (provenance in docs/logbook/altgen_fakedata.md). We apply the SAME CC0pi-Np
selection ADoNIS applies to its bank (topological 0-meson + T2K acceptance), so the
GENIE "data" and the ADoNIS prediction are compared consistently.

Selection (mirrors analysis/t2k/differentiability/tune.py _sel + bank_plot.signal_cc0pi
topological): CC event, 0 pions (nfpip=nfpim=nfpi0=0), >=1 proton; muon p>250 MeV &
cos>-0.6; leading (max-p) proton 450<p<1000 MeV & cos>0.4. Observables delta_pT,
delta_alphaT via the exact ADoNIS formulas (MeV).

Absolute normalisation: each GENIE event represents <sigma_tot>_Phi / N_gen of cross
section (per struck nucleon). We estimate <sigma_tot>_Phi two independent ways from the
per-event gst XSec [1E-38 cm^2] and cross-check: (a) harmonic mean identity, (b) flux-
reweighted sigma(E) integral. dsigma/dx in 1E-38 cm^2/(unit)/nucleon, matching the bank.

Output npz: edges, counts, dsigma (abs), and the real T2K covariance adopted as the
measurement error model (central values = GENIE).
"""
import os, sys
from pathlib import Path
import numpy as np
import uproot
import awkward as ak

GST = sys.argv[1] if len(sys.argv) > 1 else "output/altgen/genie_t2k_12C_ar23_CCQERES.gst.root"
OUT = sys.argv[2] if len(sys.argv) > 2 else "output/altgen/fakedata_ccqeres.npz"

# ---- acceptance windows (MeV) -- identical to ADoNIS tune._sel ---------------------------------- #
MU_LO, COSMU, P_LO, P_HI, COSP = 250.0, -0.6, 450.0, 1000.0, 0.4


def load_t2k(obs):
    r = uproot.open(f"../nuisance/data/T2K/CC0pi/STV/{obs}Results.root")
    if obs == "dpt":
        edges = np.asarray(r["Result"].axis().edges()) * 1000.0        # MeV
    else:
        edges = np.asarray(r["Result"].axis().edges())                 # rad
    data = np.asarray(r["Result"].values()) * 1e38
    derr = np.asarray(r["Result"].errors()) * 1e38
    cov = np.asarray(r["Covariance_Matrix"].values())
    return edges, data, derr, cov


def extract_cc0pi(GST):
    """Load a gst, apply the CC0pi-Np topological+acceptance selection, and return per-event
    observables + absolute normalisation. SINGLE SOURCE OF TRUTH for the CC0pi selection --
    reused by the fake-data builder (main) and the adaptive-binning design script."""
    print(f"[load] gst {GST}", flush=True)
    t = uproot.open(GST)["gst"]
    b = t.arrays(["cc", "qel", "res", "mec", "coh", "Ev", "XSec",
                  "pxl", "pyl", "pzl", "El", "pl", "cthl",
                  "nfp", "nfpip", "nfpim", "nfpi0", "nfkp", "nfkm", "nfk0",
                  "pdgf", "pxf", "pyf", "pzf", "pf", "Ef"], library="ak")
    N = len(b.cc)
    print(f"[load] {N} events; channel fractions: "
          f"QE={float(ak.mean(b.qel)):.3f} RES={float(ak.mean(b.res)):.3f} "
          f"MEC={float(ak.mean(b.mec)):.3f} COH={float(ak.mean(b.coh)):.3f}", flush=True)

    # ---- absolute normalisation: <sigma_tot>_Phi [1E-38 cm^2 / struck nucleon] ------------------ #
    X = ak.to_numpy(b.XSec).astype(np.float64)
    Ev = ak.to_numpy(b.Ev).astype(np.float64)
    sig_harm = N / np.sum(1.0 / X)                                      # harmonic-mean identity (XSec=total)
    # flux-reweighted: reconstruct sigma_tot(E) as the sum over ALL channels present in the sample
    # (per-channel mean XSec per E-bin; a channel absent from the run contributes zero).
    qel = ak.to_numpy(b.qel).astype(bool); res = ak.to_numpy(b.res).astype(bool)
    mec = ak.to_numpy(b.mec).astype(bool); coh = ak.to_numpy(b.coh).astype(bool)
    ebins = np.linspace(Ev.min(), min(Ev.max(), 10.0), 60)
    ec = 0.5 * (ebins[1:] + ebins[:-1])
    def sigE(mask):
        idx = np.clip(np.searchsorted(ebins, Ev) - 1, 0, len(ec) - 1)
        num = np.bincount(idx[mask], weights=X[mask], minlength=len(ec))
        cnt = np.bincount(idx[mask], minlength=len(ec))
        return np.where(cnt > 0, num / np.maximum(cnt, 1), 0.0)
    # T2K flux weight per E-bin
    fedges, fval = _flux()
    fj = np.interp(ec, 0.5 * (fedges[1:] + fedges[:-1]), fval)
    stot = sigE(qel) + sigE(res) + sigE(mec) + sigE(coh)
    sig_flux = np.sum(stot * fj) / np.sum(fj)
    print(f"[norm] <sigma_tot>_Phi  harmonic={sig_harm:.4f}  flux-reweighted={sig_flux:.4f}  "
          f"[1e-38 cm^2/nucleon]  (ratio {sig_harm/sig_flux:.3f})", flush=True)
    # Harmonic-mean identity is exact in expectation but has huge variance (dominated by tiny
    # near-threshold XSec -> 1/X outliers). The flux-reweighted sigma(E) integral bin-averages XSec
    # and is robust, so it is the absolute default. Absolute comparison is secondary; the primary
    # ADoNIS-vs-GENIE fit is SHAPE (profiled norm), which is normalisation-independent.
    SIG = sig_flux      # per-event cross section = SIG / N
    # gst XSec is the process total xsec on the NUCLEUS (12C); T2K STV + the ADoNIS bank are per
    # NUCLEON -> /12 (verified: without it GENIE lands ~10-15x above T2K data; with it ~0.9-1.3x).
    per_event = SIG / N / 12.0

    # ---- leading proton per event (max |p| among final protons) -------------------------------- #
    isp = b.pdgf == 2212
    pf_p = b.pf[isp]
    has_p = ak.num(pf_p) > 0
    # index of max-momentum proton (per event), then pick that proton's components
    jlead = ak.argmax(pf_p, axis=1, keepdims=True)
    pxp = ak.to_numpy(ak.fill_none(ak.firsts(b.pxf[isp][jlead]), 0.0))
    pyp = ak.to_numpy(ak.fill_none(ak.firsts(b.pyf[isp][jlead]), 0.0))
    pzp = ak.to_numpy(ak.fill_none(ak.firsts(b.pzf[isp][jlead]), 0.0))
    pmag = np.sqrt(pxp**2 + pyp**2 + pzp**2)
    cthp = np.where(pmag > 0, pzp / np.maximum(pmag, 1e-12), 0.0)
    hasp = ak.to_numpy(has_p)

    # ---- muon (final primary lepton), MeV ------------------------------------------------------ #
    pxl = ak.to_numpy(b.pxl) * 1000.0; pyl = ak.to_numpy(b.pyl) * 1000.0
    pzl = ak.to_numpy(b.pzl) * 1000.0
    pmu = np.sqrt(pxl**2 + pyl**2 + pzl**2); cthmu = np.where(pmu > 0, pzl / np.maximum(pmu, 1e-12), 0.0)
    pxp *= 1000.0; pyp *= 1000.0; pmag_mev = pmag * 1000.0

    # ---- CC0pi-Np topological signal + acceptance ---------------------------------------------- #
    cc = ak.to_numpy(b.cc).astype(bool)
    npi = ak.to_numpy(b.nfpip + b.nfpim + b.nfpi0)
    sig_topo = cc & (npi == 0) & hasp
    acc = (pmu > MU_LO) & (cthmu > COSMU) & (pmag_mev > P_LO) & (pmag_mev < P_HI) & (cthp > COSP)
    sel = sig_topo & acc
    print(f"[sel] CC={cc.sum()}  0pi&Np={sig_topo.sum()}  +acceptance={sel.sum()}", flush=True)

    # ---- observables (ADoNIS formulas, MeV) ---------------------------------------------------- #
    dvx = pxl + pxp; dvy = pyl + pyp
    dpt = np.sqrt(dvx**2 + dvy**2)
    num = -(pxl * dvx + pyl * dvy)
    den = np.sqrt(pxl**2 + pyl**2) * np.clip(np.sqrt(dvx**2 + dvy**2), 1e-9, None)
    dat = np.arccos(np.clip(num / den, -1.0, 1.0))

    return dict(b=b, N=N, sig_harm=sig_harm, sig_flux=sig_flux, per_event=per_event,
                sel=sel, dpt=dpt, dat=dat, cc=cc, pmu=pmu, cthmu=cthmu,
                pxl=pxl, pyl=pyl, pzl=pzl,
                chan_frac=np.array([float(ak.mean(b.qel)), float(ak.mean(b.res)),
                                    float(ak.mean(b.mec)), float(ak.mean(b.coh))]))


def extract_cc1pi(E):
    """CC1pi+Np STV selection + observables from an extract_cc0pi() dict. SINGLE SOURCE OF TRUTH --
    reused by the fake-data builder (main) and physical_fit_run mode=genie. Mirrors
    bank_plot.signal_cc1pi_stv exactly; observables via the imported validated bank_plot formulas.
    Returns vals1 {pn,dptt,daT} (selected events) + per_event_nb_CH."""
    from adonis.reweight import bank_plot as BPX
    b = E["b"]; cc = E["cc"]; pmu = E["pmu"]; cthmu = E["cthmu"]
    pxl = E["pxl"]; pyl = E["pyl"]; pzl = E["pzl"]; per_event = E["per_event"]
    CTH70 = float(np.cos(np.deg2rad(70.0)))

    npip = ak.to_numpy(b.nfpip); npi0 = ak.to_numpy(b.nfpi0); npim = ak.to_numpy(b.nfpim)
    nk = ak.to_numpy(b.nfkp + b.nfkm + b.nfk0)
    # single pi+ 4-vector (events with exactly one pi+)
    ispip = b.pdgf == 211
    pip_px = ak.to_numpy(ak.fill_none(ak.firsts(b.pxf[ispip]), 0.0)) * 1e3
    pip_py = ak.to_numpy(ak.fill_none(ak.firsts(b.pyf[ispip]), 0.0)) * 1e3
    pip_pz = ak.to_numpy(ak.fill_none(ak.firsts(b.pzf[ispip]), 0.0)) * 1e3
    pip_E  = ak.to_numpy(ak.fill_none(ak.firsts(b.Ef[ispip]), 0.0)) * 1e3
    ppi = np.sqrt(pip_px**2 + pip_py**2 + pip_pz**2)
    cpi = np.where(ppi > 0, pip_pz / np.maximum(ppi, 1e-9), 0.0)
    # leading proton WITHIN the [450,1200) window (T2K CC1pi picks leading ACCEPTED proton)
    pmev = b.pf * 1e3
    inwin = (b.pdgf == 2212) & (pmev >= 450.0) & (pmev < 1200.0)
    pf_w = b.pf[inwin]
    jw = ak.argmax(pf_w, axis=1, keepdims=True)
    lw_px = ak.to_numpy(ak.fill_none(ak.firsts(b.pxf[inwin][jw]), 0.0)) * 1e3
    lw_py = ak.to_numpy(ak.fill_none(ak.firsts(b.pyf[inwin][jw]), 0.0)) * 1e3
    lw_pz = ak.to_numpy(ak.fill_none(ak.firsts(b.pzf[inwin][jw]), 0.0)) * 1e3
    lw_E  = ak.to_numpy(ak.fill_none(ak.firsts(b.Ef[inwin][jw]), 0.0)) * 1e3
    lwp = np.sqrt(lw_px**2 + lw_py**2 + lw_pz**2)
    clw = np.where(lwp > 0, lw_pz / np.maximum(lwp, 1e-9), 0.0)
    hasw = ak.to_numpy(ak.num(pf_w) > 0)
    Emu = ak.to_numpy(b.El) * 1e3
    sel1 = (cc & (npip == 1) & (npi0 == 0) & (npim == 0) & (nk == 0) & hasw
            & (pmu >= 250.0) & (pmu < 7000.0) & (cthmu > CTH70)
            & (ppi >= 150.0) & (ppi < 1200.0) & (cpi > CTH70) & (clw > CTH70))
    print(f"\n[sel CC1pi] 1pi+(no other meson)={int((cc&(npip==1)&(npi0==0)&(npim==0)&(nk==0)).sum())}"
          f"  +p-window={int((cc&(npip==1)&(npi0==0)&(npim==0)&(nk==0)&hasw).sum())}"
          f"  +acceptance={int(sel1.sum())}", flush=True)
    kmu4 = np.stack([Emu, pxl, pyl, pzl], axis=1)[sel1]
    lead4 = np.stack([lw_E, lw_px, lw_py, lw_pz], axis=1)[sel1]
    pip4 = np.stack([pip_E, pip_px, pip_py, pip_pz], axis=1)[sel1]
    vals1 = {"pn":   np.asarray(BPX.pN_1pi(kmu4, lead4, pip4)),
             "dptt": np.asarray(BPX.dptt_1pi(kmu4, lead4, pip4)),
             "daT":  np.degrees(np.asarray(BPX.dat_1pi(kmu4, lead4, pip4)))}
    per_event_nb_CH = per_event * 12.0 * 1e-5      # 1e-38 cm^2/nucleon -> nb per C(==C-part of CH)
    # pion kinematics of the SELECTED events (for the extended physfit suite)
    return dict(sel1=sel1, vals1=vals1, per_event_nb_CH=per_event_nb_CH,
                ppi=ppi[sel1], cospi=cpi[sel1])


def main():
    E = extract_cc0pi(GST)
    b = E["b"]; N = E["N"]; per_event = E["per_event"]; sel = E["sel"]
    dpt = E["dpt"]; dat = E["dat"]; sig_harm = E["sig_harm"]; sig_flux = E["sig_flux"]
    cc = E["cc"]; pmu = E["pmu"]; cthmu = E["cthmu"]
    pxl = E["pxl"]; pyl = E["pyl"]; pzl = E["pzl"]
    out = {"gst": GST, "N_gen": N, "sig_tot_harm": sig_harm, "sig_tot_flux": sig_flux,
           "chan_frac": E["chan_frac"]}
    for obs, val in (("dpt", dpt), ("dat", dat)):
        edges, tdata, tderr, tcov = load_t2k(obs)
        v = val[sel]
        cnt, _ = np.histogram(v, bins=edges)
        bw_unit = np.diff(edges) / (1000.0 if obs == "dpt" else 1.0)     # GeV/c or rad
        dsig = cnt * per_event / bw_unit                                 # 1e-38 cm^2/unit/nucleon
        # MC stat error on the fake data central value (Poisson counts)
        dsig_err = np.sqrt(cnt) * per_event / bw_unit
        out[f"{obs}_edges"] = edges; out[f"{obs}_counts"] = cnt
        out[f"{obs}_dsig"] = dsig; out[f"{obs}_dsig_mcerr"] = dsig_err
        out[f"{obs}_t2k_data"] = tdata; out[f"{obs}_t2k_derr"] = tderr; out[f"{obs}_t2k_cov"] = tcov
        print(f"\n==== {obs}: GENIE fake dsigma vs T2K real data (1e-38 units) ====")
        print(f"{'bin':>20} {'count':>8} {'GENIE':>10} {'T2Kdata':>10} {'ratio':>7}")
        for i in range(len(cnt)):
            print(f"[{edges[i]:8.3f},{edges[i+1]:8.3f}] {cnt[i]:8d} {dsig[i]:10.4f} "
                  f"{tdata[i]:10.4f} {dsig[i]/max(tdata[i],1e-9):7.3f}")
    # ================= CC1pi+Np STV fake datasets (pN, dpTT, daT) =============================== #
    # Selection + observables factored into extract_cc1pi (single source of truth). Units: nb/unit
    # per CH -- GENIE-C part only; the frozen ADoNIS free-H offset is added by the fit (identically
    # to the model, so H cancels in residuals). load_cc1pi gives T2K edges/cov in nb/CH.
    from adonis.measurements.t2k_stv import load_cc1pi
    E1 = extract_cc1pi(E)
    vals1 = E1["vals1"]; per_event_nb_CH = E1["per_event_nb_CH"]
    NAME1 = {"pn": "pN", "dptt": "dpTT", "daT": "daT"}
    for dkey, v in vals1.items():
        edges, tdata, tcov = load_cc1pi(NAME1[dkey])
        cnt, _ = np.histogram(v, bins=edges)
        dsig = cnt * per_event_nb_CH / np.diff(edges)              # nb/unit/CH, GENIE-C only (NO free-H)
        out[f"cc1pi_{dkey}_edges"] = edges; out[f"cc1pi_{dkey}_counts"] = cnt
        out[f"cc1pi_{dkey}_dsig_C"] = dsig
        out[f"cc1pi_{dkey}_t2k_data"] = tdata; out[f"cc1pi_{dkey}_t2k_cov"] = tcov
        print(f"==== CC1pi {NAME1[dkey]}: GENIE-C dsig (nb/unit/CH, no free-H) vs T2K data ====")
        for i in range(len(cnt)):
            print(f"[{edges[i]:8.2f},{edges[i+1]:8.2f}] {cnt[i]:7d} {dsig[i]:10.5f} {tdata[i]:10.5f} "
                  f"{dsig[i]/max(tdata[i],1e-12):7.3f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **out)
    print(f"\n[out] {OUT}", flush=True)


def _flux():
    p = "output/altgen/T2K_nu.dat"
    elo, ehi, val = [], [], []
    for line in open(p):
        q = line.split()
        if len(q) != 4:
            continue
        try:
            elo.append(float(q[1])); ehi.append(float(q[2])); val.append(float(q[3]))
        except ValueError:
            continue
    edges = np.array(elo + [ehi[-1]]); return edges, np.array(val)


if __name__ == "__main__":
    main()
