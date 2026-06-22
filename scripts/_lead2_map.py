"""Map the post-lead#2 residuals: decompose the absorbed-RES (CC0pi.RES) rate deficit into
RES total-sigma vs absorption-fraction, and dump the absorbed-RES leading-proton spectrum
(0p residual).  ADoNIS lead2 (high-cap, neutrons re-cascaded) vs ACHILLES-C FSI proc 401/402."""
import os, sys, math, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, yaml
from adonis.workflow.config import load_analysis_config
import adonis.workflow.analyze as A
from scripts.gen_cc_matrix import signal_block, banks, neff, _obs

REF = "data/oracle/t2k_cc1pi_rich_ach_FSI_proc.npz"
QE  = "data/oracle/t2k_cc0pi_engine_rich_cv5.npz"
RB  = "data/oracle/t2k_cc1pi_engine_rich_lead2.npz"

def cell(channel, pcat):
    sigd, obs = signal_block(channel, "res", pcat)
    inp = banks("res", RB, QE); inp["reference"] = [REF]
    cfg_d = dict(inputs=inp, signal=sigd, observables=_obs(obs), data={"enabled": False},
                 out_path="/tmp/_map.png", carbon_only=True)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg_d, fh); tmp = fh.name
    cfg = load_analysis_config(tmp); os.unlink(tmp)
    ado = A.build_adonis(cfg); ref = A.build_reference(cfg)
    Na, nea, sa = neff(ado["w"]); Nr, ner, sr = neff(ref["w"])
    pa = (ado["p_lead"] if "p_lead" in ado else None)
    return sa, sr, Na, Nr, nea, ner, ado, ref

print("== RES rate decomposition: absorbed (CC0pi) vs pion-survived (CC1pi) ==")
print(f"{'channel':8s} {'pcat':5s} {'sig_ADO':>11s} {'sig_ACH':>11s} {'ACH/ADO':>9s} {'pull':>6s}")
tot = {}
for channel in ("cc0pi", "cc1pi"):
    sa, sr, Na, Nr, nea, ner, ado, ref = cell(channel, "incl")
    # add 0p (no-proton) to get the full channel total
    sa0, sr0, *_ = cell(channel, "0p")
    sat, srt = sa + sa0, sr + sr0
    tot[channel] = (sat, srt)
    ratio = sr/sa if sa else float('nan'); rel = math.sqrt((1/nea if nea else 0)+(1/ner if ner else 0))
    pull = (ratio-1)/(ratio*rel) if rel else float('nan')
    print(f"{channel:8s} {'all':5s} {sat:11.4e} {srt:11.4e} {srt/sat:9.3f} {'':6s}  (incl ratio {ratio:.3f} {pull:+.1f}s)")

ca, cr = tot["cc0pi"]; pa, pr = tot["cc1pi"]
print(f"\nRES total (CC0pi+CC1pi):  ADO={ca+pa:.4e}  ACH={cr+pr:.4e}  ACH/ADO={(cr+pr)/(ca+pa):.3f}")
print(f"absorbed fraction CC0pi/(CC0pi+CC1pi):  ADO={ca/(ca+pa):.4f}  ACH={cr/(cr+pr):.4f}")
print(f"  -> if RES-total matches but abs-frac differs => absorption-fraction lead; else sigma lead")

# absorbed-RES leading-proton spectrum, threshold lowered to 0 (0p residual: abs protons too soft?)
print("\n== absorbed-RES (CC0pi) leading-proton |p| spectrum, eject_thresh=0 (frac of events) ==")
def cell_lp(eject):
    sigd, obs = signal_block("cc0pi", "res", "incl")
    sigd["eject_thresh"] = eject; sigd["proton_lead"] = "global"
    sigd["p_win"] = [1.0, "inf"]; sigd["require_proton"] = True  # >=1 proton of ANY momentum
    sigd["proton_count"] = "ge1"; sigd["cth"] = -1.0             # drop forward cut to see full spectrum
    inp = banks("res", RB, QE); inp["reference"] = [REF]
    cfg_d = dict(inputs=inp, signal=sigd,
                 observables=_obs([("lp_p", "lead p", ("lin", 0, 1200, 13))]),
                 data={"enabled": False}, out_path="/tmp/_map.png", carbon_only=True)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg_d, fh); tmp = fh.name
    cfg = load_analysis_config(tmp); os.unlink(tmp)
    ado = A.build_adonis(cfg); ref = A.build_reference(cfg)
    return ado, ref
ado, ref = cell_lp(0.0)
edges = [0, 150, 250, 350, 450, 600, 900, 1200]
la, wa = ado["lp_p"], ado["w"]; lr, wr = ref["lp_p"], ref["w"]
fa = wa.sum(); fr = wr.sum()
print(f"{'bracket':12s} {'frac_ADO':>9s} {'frac_ACH':>9s} {'ACH/ADO':>9s}")
for lo, hi in zip(edges[:-1], edges[1:]):
    ma = (la >= lo) & (la < hi); mr = (lr >= lo) & (lr < hi)
    pa_ = wa[ma].sum()/fa; pr_ = wr[mr].sum()/fr
    print(f"[{lo:4d},{hi:4d}) {pa_:9.4f} {pr_:9.4f} {(pr_/pa_) if pa_ else float('nan'):9.3f}")
