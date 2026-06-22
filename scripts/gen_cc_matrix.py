"""Generate the full CC0pi/CC1pi x {QE, RES, QE+RES} x {incl, 0p, 1p, 2p} x {C, Ar} figure matrix,
ADoNIS vs ACHILLES (ratio + chi2), and a summary table with selected sigma, N_eff (both sides) and
the significance of each ACH/ADO ratio.

Proton multiplicity conventions (each the standard for its measurement, both wired on both sides):
  CC0pi -> ejected protons >eject_thresh (n_ejected), lead = global   [T2K CC0pi TKI convention]
  CC1pi -> in-window protons in p_win (proton_count eq0/eq1/eq2)       [CC1pi+Np convention]
'incl' = the +Np inclusive (>=1 proton).  0p = exactly 0; 1p/2p = exactly 1/2.

H / CH are NOT produced: the free-proton (adonis_h) combination is not wired in build_adonis and there
are no H banks / H reference.  This script covers C and Ar.

Usage: python -u scripts/gen_cc_matrix.py [C|Ar|all]
"""
import os, sys, math, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, yaml
from adonis.workflow.config import load_analysis_config
import adonis.workflow.analyze as A
import adonis.workflow.signal as SG

def _pick(*cands):
    """First existing path among candidates (high-seed bank, else fall back to 5-seed)."""
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[-1]

# material -> (RES bank, QE bank, ACHILLES proc-tagged reference); prefer high-seed, fall back to cv5/av5
MATERIALS = {
    "C":  (_pick("data/oracle/t2k_cc1pi_engine_rich_cv12.npz", "data/oracle/t2k_cc1pi_engine_rich_cv5.npz"),
           _pick("data/oracle/t2k_cc0pi_engine_rich_cv30.npz", "data/oracle/t2k_cc0pi_engine_rich_cv5.npz"),
           "data/oracle/t2k_cc1pi_rich_ach_FSI_proc.npz"),
    "Ar": (_pick("data/oracle/t2k_cc1pi_engine_rich_av12.npz", "data/oracle/t2k_cc1pi_engine_rich_av5.npz"),
           _pick("data/oracle/t2k_cc0pi_engine_rich_av30.npz", "data/oracle/t2k_cc0pi_engine_rich_av5.npz"),
           "data/oracle/t2k_cc1pi_rich_ach_FSI_Ar_proc.npz"),
}
REF_PROC = {"qe": [200], "res": [401, 402], "both": None}

# PRE-FSI (no cascade) bank set + PURE-target ACHILLES no-FSI references (no proc field needed:
# pre-FSI the pion selection itself separates channels -- CC0pi=pion-veto->QE, CC1pi=pi+ ->RES).
MATERIALS_NOFSI = {
    "C":  ("data/oracle/t2k_cc1pi_engine_rich_nofsi.npz", "data/oracle/t2k_cc0pi_engine_rich_nofsi.npz",
           "data/oracle/t2k_cc1pi_rich_ach_C_nofsi.npz"),
    "Ar": ("data/oracle/t2k_cc1pi_engine_rich_ar_nofsi.npz", "data/oracle/t2k_cc0pi_engine_rich_ar_nofsi.npz",
           "data/oracle/t2k_cc1pi_rich_ach_nofsi_Ar.npz"),
}

# observable panels
TKI   = [("pn", "$p_N$", [0, 120, 240, 600, 1500]),
         ("dptt", r"$\delta p_{TT}$", [-700, -300, -100, 100, 300, 700]),
         ("dalphat", r"$\delta\alpha_T$", ("lin", 0, "pi", 7)),
         ("W", "$W$", ("lin", 1080, 1700, 13)), ("Q2", "$Q^2$", ("lin", 0, 1.5, 13)),
         ("pi_p", r"$p_\pi$", ("lin", 150, 1200, 13)), ("lp_p", "lead $p_p$", ("lin", 450, 1200, 13))]
PIFREE = [("W", "$W$", ("lin", 1080, 1700, 13)), ("Q2", "$Q^2$", ("lin", 0, 1.5, 13)),
          ("pi_p", r"$p_\pi$", ("lin", 150, 1200, 13)), ("p_mu", r"$p_\mu$", ("lin", 0, 2000, 21)),
          ("cos_mu", r"$\cos\theta_\mu$", ("lin", -0.6, 1.0, 21))]
CC0   = [("dpt", r"$\delta p_T$", ("lin", 0, 800, 21)), ("dalphat", r"$\delta\alpha_T$", ("lin", 0, "pi", 21)),
         ("Q2", "$Q^2$", ("lin", 0, 1.4, 21)), ("W", "$W$", ("lin", 938, 1400, 21)),
         ("p_mu", r"$p_\mu$", ("lin", 0, 2000, 21)), ("cos_mu", r"$\cos\theta_\mu$", ("lin", -0.6, 1.0, 21)),
         ("lp_p", "lead $p_p$", ("lin", 450, 1000, 21))]
MUON  = [("Q2", "$Q^2$", ("lin", 0, 1.4, 21)), ("p_mu", r"$p_\mu$", ("lin", 0, 2000, 21)),
         ("cos_mu", r"$\cos\theta_\mu$", ("lin", -0.6, 1.0, 21))]

def _obs(spec):
    out = []
    for key, label, b in spec:
        if isinstance(b, tuple) and b[0] == "lin":
            out.append({"key": key, "label": label, "linspace": list(b[1:])})
        else:
            out.append({"key": key, "label": label, "edges": b})
    return out

def signal_block(channel, contrib, pcat):
    """Return (signal dict, observable spec) for one cell."""
    if channel == "cc0pi":
        sd = dict(mu_win=[250.0, "inf"], cos_mu=-0.6, p_win=[450.0, 1000.0], cth=0.4, pi_win=None,
                  proton_lead="global", pion_id="none", eject_thresh=250.0, target="carbon")
        sd["ref_proc"] = REF_PROC[contrib]
        if pcat == "incl":
            sd.update(proton_count="ge1", require_proton=True); return sd, CC0
        if pcat == "0p":
            sd.update(require_proton=False, n_ejected=0); return sd, MUON
        sd.update(require_proton=True, n_ejected=int(pcat[0])); return sd, CC0
    # cc1pi
    sd = dict(mu_win=[250.0, 7000.0], pi_win=[150.0, 1200.0], p_win=[450.0, 1200.0], cth="cos70",
              proton_lead="in_window", pion_id="pip", count_recoil_neutron=False, target="carbon",
              W_conv="vertex", ref_proc=REF_PROC[contrib])
    if pcat == "incl":
        sd.update(proton_count="ge1", require_proton=True); return sd, TKI
    if pcat == "0p":
        sd.update(proton_count="eq0", require_proton=False); return sd, PIFREE
    sd.update(proton_count="eq" + pcat[0], require_proton=True); return sd, TKI

def banks(contrib, res_bank, qe_bank):
    if contrib == "qe":  return {"adonis_qe": [qe_bank]}
    if contrib == "res": return {"adonis_res": [res_bank]}
    return {"adonis_res": [res_bank], "adonis_qe": [qe_bank]}

def neff(w):
    w = np.asarray(w); w = w[w > 0]
    return (int(w.size), float(w.sum()**2 / np.sum(w**2)) if w.size else 0.0, float(w.sum()))

def run_cell(material, channel, contrib, pcat, mode="fsi"):
    mats = MATERIALS_NOFSI if mode == "nofsi" else MATERIALS
    res_bank, qe_bank, ref = mats[material]
    if mode == "nofsi" and (not os.path.exists(res_bank) or not os.path.exists(qe_bank) or not os.path.exists(ref)):
        return dict(cell=f"{channel}_{contrib}_{pcat}_{material}", ok=False, err="bank/ref missing")
    sigd, obs = signal_block(channel, contrib, pcat)
    if mode == "nofsi":
        sigd["ref_proc"] = None                     # pre-FSI: selection separates channels, no proc filter
    outdir = "paper_figures/matrix_nofsi" if mode == "nofsi" else "paper_figures/matrix"
    inp = banks(contrib, res_bank, qe_bank); inp["reference"] = [ref]
    tag = f"{channel}_{contrib}_{pcat}_{material}"
    cfg_d = dict(inputs=inp, signal=sigd, observables=_obs(obs), data={"enabled": False},
                 out_path=f"{outdir}/{tag}.png", carbon_only=True, ratio_ylim=[0.5, 1.6],
                 title=f"{material} {channel} [{contrib}] {pcat} {'noFSI' if mode=='nofsi' else 'FSI'}: ADoNIS vs ACHILLES")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg_d, fh); tmp = fh.name
    cfg = load_analysis_config(tmp); os.unlink(tmp)
    try:
        ado = A.build_adonis(cfg); ref_d = A.build_reference(cfg)
        Na, nea, sa = neff(ado["w"]); Nr, ner, sr = neff(ref_d["w"])
        if Na == 0 or Nr == 0:                      # empty cell (e.g. pre-FSI CC0pi-RES / CC1pi-QE)
            return dict(cell=tag, ok=False, err="empty selection")
        A.run_analysis(cfg)
    except Exception as e:
        return dict(cell=tag, ok=False, err=str(e)[:80])
    ratio = sr / sa if sa else float("nan")
    rel = math.sqrt((1/nea if nea else 0) + (1/ner if ner else 0))
    pull = (ratio - 1) / (ratio * rel) if (rel and ratio == ratio) else float("nan")
    return dict(cell=tag, ok=True, sa=sa, sr=sr, ratio=ratio, Na=Na, nea=nea, Nr=Nr, ner=ner,
                err=ratio*rel, pull=pull)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    mode = sys.argv[2] if len(sys.argv) > 2 else "fsi"
    os.makedirs("paper_figures/matrix_nofsi" if mode == "nofsi" else "paper_figures/matrix", exist_ok=True)
    base = MATERIALS_NOFSI if mode == "nofsi" else MATERIALS
    mats = list(base) if which == "all" else [which]
    contribs = ("both",) if mode == "nofsi" else ("qe", "res", "both")  # pre-FSI: selection separates channels
    rows = []
    for material in mats:
        for channel in ("cc0pi", "cc1pi"):
            for contrib in contribs:
                for pcat in ("incl", "0p", "1p", "2p"):
                    r = run_cell(material, channel, contrib, pcat, mode)
                    rows.append(r)
                    if r["ok"]:
                        print(f"[OK] {r['cell']:26s} ACH/ADO={r['ratio']:.3f}+/-{r['err']:.3f} "
                              f"({r['pull']:+.1f}s)  N_eff ado={r['nea']:.0f} ach={r['ner']:.0f}", flush=True)
                    else:
                        print(f"[FAIL] {r['cell']:26s} {r['err']}", flush=True)
    # summary table
    print("\n==== SUMMARY ====")
    print(f"{'cell':30s} {'ACH/ADO':>9s} {'pull':>6s} {'Neff_ado':>9s} {'Neff_ach':>9s}")
    for r in rows:
        if r["ok"]:
            print(f"{r['cell']:30s} {r['ratio']:9.3f} {r['pull']:+6.1f} {r['nea']:9.0f} {r['ner']:9.0f}")
        else:
            print(f"{r['cell']:30s}   FAIL: {r['err']}")
    np.savez(f"/tmp/cc_matrix_summary_{mode}.npz", rows=rows)
    print("MATRIX DONE", flush=True)
