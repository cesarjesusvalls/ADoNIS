"""Lead #2 check: CC0pi.RES (absorbed pion -> 0pi) proton-multiplicity cross sections,
NEW bank (re-cascade abs neutrons) vs OLD bank (proton-only abs slots) vs ACHILLES-C.
Reuses the exact signal machinery from gen_cc_matrix (apples-to-apples both sides)."""
import os, sys, math, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, yaml
from adonis.workflow.config import load_analysis_config
import adonis.workflow.analyze as A
from scripts.gen_cc_matrix import signal_block, banks, neff, _obs, REF_PROC

REF = "data/oracle/t2k_cc1pi_rich_ach_FSI_proc.npz"
QE  = "data/oracle/t2k_cc0pi_engine_rich_cv5.npz"   # QE bank (shared; contrib=res so unused)

def cell(res_bank, pcat):
    sigd, obs = signal_block("cc0pi", "res", pcat)
    inp = banks("res", res_bank, QE); inp["reference"] = [REF]
    cfg_d = dict(inputs=inp, signal=sigd, observables=_obs(obs), data={"enabled": False},
                 out_path="/tmp/_lead2.png", carbon_only=True)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg_d, fh); tmp = fh.name
    cfg = load_analysis_config(tmp); os.unlink(tmp)
    ado = A.build_adonis(cfg); ref = A.build_reference(cfg)
    Na, nea, sa = neff(ado["w"]); Nr, ner, sr = neff(ref["w"])
    return sa, sr, Na, Nr, nea, ner

banks_test = [("OLD cv5", "data/oracle/t2k_cc1pi_engine_rich_cv5.npz"),
              ("OLD hicap", "data/oracle/t2k_cc1pi_engine_rich_hicap.npz"),
              ("NEW lead2", "data/oracle/t2k_cc1pi_engine_rich_lead2.npz")]
# ACHILLES absolute per multiplicity (from any cell's sr -- same ref)
print(f"{'bank':12s} {'pcat':5s} {'sig_ADO':>11s} {'sig_ACH':>11s} {'ACH/ADO':>9s} {'pull':>6s}  N_ado N_ach")
for name, rb in banks_test:
    if not os.path.exists(rb):
        print(f"{name}: bank missing {rb}"); continue
    for pcat in ("incl", "0p", "1p", "2p"):
        sa, sr, Na, Nr, nea, ner = cell(rb, pcat)
        ratio = sr/sa if sa else float("nan")
        rel = math.sqrt((1/nea if nea else 0)+(1/ner if ner else 0))
        pull = (ratio-1)/(ratio*rel) if rel else float("nan")
        print(f"{name:12s} {pcat:5s} {sa:11.4e} {sr:11.4e} {ratio:9.3f} {pull:+6.1f}  {Na:6d} {Nr:6d}")
