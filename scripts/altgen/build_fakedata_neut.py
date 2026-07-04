"""Build T2K-like CC0pi + CC1pi STV fake datasets from a NEUT (nuisflat GenericVectors) flat tree.

Same structure as build_fakedata.py (GENIE): ADoNIS's exact selections + T2K binning/covariance,
central values from the foreign generator. Two channel slices from ONE NEUT run:
  * matched  : Mode in {1 (CCQE), 11,12,13 (CC1pi)} -- the ADoNIS channel set (QE+RES analogue)
  * full     : all CC modes (incl. 2p2h Mode 2, multi-pi 21, DIS 26, coh 16) -- contains physics
               ADoNIS does not model at all (the strongest unmodelled-physics probe)
Normalisation: NUISANCE fScaleFactor (per-event, converts counts to dsigma; units verified against
the T2K data scale at build time -- printed).
Output npz keys mirror build_fakedata.py (dpt/dat + cc1pi_{pn,dptt,daT}) with a '_full' suffix set.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import uproot
import awkward as ak

FLAT = sys.argv[1] if len(sys.argv) > 1 else "output/altgen/neut_t2k_12C_flat.root"
OUT = sys.argv[2] if len(sys.argv) > 2 else "output/altgen/fakedata_neut.npz"

MU_LO, COSMU, P_LO, P_HI, COSP = 250.0, -0.6, 450.0, 1000.0, 0.4
CTH70 = float(np.cos(np.deg2rad(70.0)))
MATCHED = (1, 11, 12, 13)                       # CCQE + CC1pi (RES) NEUT modes


def load_t2k(obs):
    r = uproot.open(f"../nuisance/data/T2K/CC0pi/STV/{obs}Results.root")
    edges = np.asarray(r["Result"].axis().edges()) * (1000.0 if obs == "dpt" else 1.0)
    data = np.asarray(r["Result"].values()) * 1e38
    cov = np.asarray(r["Covariance_Matrix"].values())
    return edges, data, cov


def main():
    print(f"[load] {FLAT}", flush=True)
    t = uproot.open(FLAT)["FlatTree_VARS"]
    b = t.arrays(["Mode", "cc", "PDGnu", "fScaleFactor", "pdg", "px", "py", "pz", "E"], library="ak")
    N = len(b.cc)
    cc = ak.to_numpy(b.cc) == 1
    mode = ak.to_numpy(b.Mode)
    sf = ak.to_numpy(b.fScaleFactor).astype(np.float64)
    import collections
    print(f"[load] {N} events; CC={cc.sum()}; CC mode composition: "
          f"{dict(sorted(collections.Counter(mode[cc]).items()))}", flush=True)
    print(f"[norm] fScaleFactor: {sf.min():.3e} .. {sf.max():.3e} (const={np.allclose(sf, sf[0])})", flush=True)

    # ---- final-state particles (GeV -> MeV) ----------------------------------------------------- #
    ispro = b.pdg == 2212
    ispip = b.pdg == 211
    ispi0 = b.pdg == 111
    ispim = b.pdg == -211
    ismu = b.pdg == 13
    isk = (b.pdg == 321) | (b.pdg == -321) | (b.pdg == 311) | (b.pdg == -311) | (b.pdg == 130) | (b.pdg == 310)
    npro = ak.to_numpy(ak.num(b.pdg[ispro]))
    npip = ak.to_numpy(ak.num(b.pdg[ispip]))
    npi0 = ak.to_numpy(ak.num(b.pdg[ispi0]))
    npim = ak.to_numpy(ak.num(b.pdg[ispim]))
    nk = ak.to_numpy(ak.num(b.pdg[isk]))

    def first4(m):
        """(E,px,py,pz) MeV of the first particle matching mask m (0 if none)."""
        return tuple(ak.to_numpy(ak.fill_none(ak.firsts(c[m]), 0.0)) * 1e3
                     for c in (b.E, b.px, b.py, b.pz))

    def lead4(m):
        """(E,px,py,pz) MeV of the max-|p| particle matching mask m (0 if none)."""
        pm = np.sqrt((b.px[m]) ** 2 + (b.py[m]) ** 2 + (b.pz[m]) ** 2)
        j = ak.argmax(pm, axis=1, keepdims=True)
        return tuple(ak.to_numpy(ak.fill_none(ak.firsts(c[m][j]), 0.0)) * 1e3
                     for c in (b.E, b.px, b.py, b.pz))

    muE, mux, muy, muz = first4(ismu)
    pmu = np.sqrt(mux**2 + muy**2 + muz**2); cmu = np.where(pmu > 0, muz / np.maximum(pmu, 1e-9), 0.0)
    # CC0pi leading proton: global max-p proton
    lE, lx, ly, lz = lead4(ispro)
    lp = np.sqrt(lx**2 + ly**2 + lz**2); cl = np.where(lp > 0, lz / np.maximum(lp, 1e-9), 0.0)
    # CC1pi: leading proton WITHIN [450,1200) + the single pi+
    pmev = np.sqrt(b.px**2 + b.py**2 + b.pz**2) * 1e3
    inwin = ispro & (pmev >= 450.0) & (pmev < 1200.0)
    wE, wx, wy, wz = lead4(inwin)
    wp = np.sqrt(wx**2 + wy**2 + wz**2); cw = np.where(wp > 0, wz / np.maximum(wp, 1e-9), 0.0)
    hasw = ak.to_numpy(ak.num(b.pdg[inwin])) > 0
    piE, pix, piy, piz = first4(ispip)
    ppi = np.sqrt(pix**2 + piy**2 + piz**2); cpi = np.where(ppi > 0, piz / np.maximum(ppi, 1e-9), 0.0)

    from analysis.t2k.differentiability import bank_plot as BPX
    from analysis.t2k.differentiability.info_content import load_cc1pi

    out = {"flat": FLAT, "N_gen": N}
    for tag, chmask in (("", np.isin(mode, MATCHED)), ("_full", np.ones(N, bool))):
        base = cc & (ak.to_numpy(b.PDGnu) == 14) & chmask
        # ---- CC0pi-Np ---------------------------------------------------------------------------- #
        sel0 = (base & (npip == 0) & (npi0 == 0) & (npim == 0) & (npro >= 1)
                & (pmu > MU_LO) & (cmu > COSMU) & (lp > P_LO) & (lp < P_HI) & (cl > COSP))
        dvx = mux + lx; dvy = muy + ly
        dpt = np.sqrt(dvx**2 + dvy**2)
        num = -(mux * dvx + muy * dvy)
        den = np.sqrt(mux**2 + muy**2) * np.clip(np.sqrt(dvx**2 + dvy**2), 1e-9, None)
        dat = np.arccos(np.clip(num / den, -1.0, 1.0))
        print(f"\n[sel{tag or ' matched'} CC0pi] {int(sel0.sum())} events", flush=True)
        for obs, val in (("dpt", dpt), ("dat", dat)):
            edges, tdata, tcov = load_t2k(obs)
            bw = np.diff(edges) / (1000.0 if obs == "dpt" else 1.0)
            hw, _ = np.histogram(val[sel0], bins=edges, weights=sf[sel0])
            dsig = hw / bw * 1e38                                    # fScaleFactor cm^2 -> 1e-38 units
            out[f"{obs}{tag}_edges"] = edges; out[f"{obs}{tag}_dsig"] = dsig
            out[f"{obs}{tag}_t2k_data"] = tdata; out[f"{obs}{tag}_t2k_cov"] = tcov
            print(f"  {obs}: NEUT/T2K ratios: " +
                  " ".join(f"{dsig[i]/max(tdata[i],1e-9):.2f}" for i in range(len(tdata))), flush=True)
        # ---- CC1pi+Np STV ------------------------------------------------------------------------ #
        sel1 = (base & (npip == 1) & (npi0 == 0) & (npim == 0) & (nk == 0) & hasw
                & (pmu >= 250.0) & (pmu < 7000.0) & (cmu > CTH70)
                & (ppi >= 150.0) & (ppi < 1200.0) & (cpi > CTH70) & (cw > CTH70))
        print(f"[sel{tag or ' matched'} CC1pi] {int(sel1.sum())} events", flush=True)
        kmu4 = np.stack([muE, mux, muy, muz], 1)[sel1]
        ld4 = np.stack([wE, wx, wy, wz], 1)[sel1]
        pp4 = np.stack([piE, pix, piy, piz], 1)[sel1]
        vals1 = {"pn": np.asarray(BPX.pN_1pi(kmu4, ld4, pp4)),
                 "dptt": np.asarray(BPX.dptt_1pi(kmu4, ld4, pp4)),
                 "daT": np.degrees(np.asarray(BPX.dat_1pi(kmu4, ld4, pp4)))}
        NAME1 = {"pn": "pN", "dptt": "dpTT", "daT": "daT"}
        w1 = sf[sel1]
        for dkey, v in vals1.items():
            edges, tdata, tcov = load_cc1pi(NAME1[dkey])
            hw, _ = np.histogram(v, bins=edges, weights=w1)
            # fScaleFactor cm^2/nucleon -> nb/CH C-part: x1e33 (nb) x12 (C nucleus)
            dsig = hw / np.diff(edges) * 1e33 * 12.0
            out[f"cc1pi_{dkey}{tag}_edges"] = edges; out[f"cc1pi_{dkey}{tag}_dsig_C"] = dsig
            out[f"cc1pi_{dkey}{tag}_t2k_data"] = tdata; out[f"cc1pi_{dkey}{tag}_t2k_cov"] = tcov
            print(f"  CC1pi {NAME1[dkey]}: NEUT-C/T2K(CH) ratios: " +
                  " ".join(f"{dsig[i]/max(tdata[i],1e-12):.2f}" for i in range(len(tdata))), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **out)
    print(f"\n[out] {OUT}", flush=True)


if __name__ == "__main__":
    main()
