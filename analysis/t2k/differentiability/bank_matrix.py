"""Run the T2K ADoNIS-vs-ACHILLES MATRIX with the ADoNIS side sourced from the differentiable EVENT BANK
(the same 1M events as the variation plots), instead of the separate forward banks.

The forward 'qe_bank'/'res_bank' are the QE-generated / RES-generated samples == the event bank's
channel==0 / channel==1.  signal.py needs only the rich-schema fields {mu,nu,struck,w,prot,pi_post,cr_p4,
cr_pid,pid_pi}, all derivable from the bank's ragged final state.  So: split the bank by channel, write
each as a rich-schema npz, point ADONIS_QE_BANK/ADONIS_RES_BANK at them, and run make_plots --matrix
UNCHANGED (validated cell/acceptance/observable logic; ACHILLES oracle as-is).

  python analysis/t2k/differentiability/bank_matrix.py [bankdir]      # -> output/figures/t2k_plots.pdf
"""
import os, sys, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import analysis.t2k.differentiability.bank_plot as BP

_PIONS = (211, 111, -211)


def _topk(B, partmask, K):
    """(p4 (n,K,4), pid (n,K)) of the top-K |p| particles (per event) selected by partmask(fs_pid)."""
    pid = B["fs_pid"]; p4 = B["fs_p4"]; eidx = B["_eidx"]; n = len(B["w0"])
    sel = partmask(pid)
    ev = eidx[sel]; pp = p4[sel].astype(np.float64); pid_s = pid[sel]
    o4 = np.zeros((n, K, 4), np.float32); opid = np.zeros((n, K), np.int32)
    if ev.size:
        mom = np.linalg.norm(pp[:, 1:], axis=1)
        order = np.lexsort((-mom, ev)); ev = ev[order]; pp = pp[order]; pid_s = pid_s[order]
        newg = np.r_[True, ev[1:] != ev[:-1]]
        gstart = np.maximum.accumulate(np.where(newg, np.arange(ev.size), 0))
        rank = np.arange(ev.size) - gstart; keep = rank < K
        o4[ev[keep], rank[keep]] = pp[keep]; opid[ev[keep], rank[keep]] = pid_s[keep]
    return o4, opid


def schema(B, ch, outpath):
    s = B["channel"] == ch
    prot, _ = _topk(B, lambda pid: pid == 2212, 4)
    pip4, pipid = _topk(B, lambda pid: np.isin(pid, _PIONS), 2)
    d = dict(mu=B["k_mu"][s].astype(np.float32), nu=B["k_nu"][s].astype(np.float32),
             struck=B["p_struck"][s].astype(np.float32), w=B["w0"][s].astype(np.float64),
             prot=prot[s], pi_post=pip4[s, 0], cr_p4=pip4[s, 1],
             pid_pi=pipid[s, 0].astype(np.int32), cr_pid=pipid[s, 1].astype(np.int32))
    np.savez(outpath, **d)
    return int(s.sum())


_PMIN = 1e-6   # |p| floor: drop recaptured/rest nucleons, matching ACHILLES + generate.py (alN & |p|>0)


def _mult_count(B, pidval):
    mom = np.linalg.norm(B["fs_p4"][:, 1:], axis=1)
    return BP._event_sum(B, ((B["fs_pid"] == pidval) & (mom > _PMIN)).astype(float)).astype(int)


def export_mult_batches(B, material, outdir="output/adonis", tag="_evb"):
    """Write the per-channel multiplicity + nucleon-rank ADoNIS batches (make_plots block schema) FROM THE
    EVENT BANK, so block_multiplicity / block_nucleon_momentum source the SAME 1M events as the matrix +
    arrows.  Final-state counts require |p|>0 (rest/recaptured nucleons excluded) to match ACHILLES.
    Files: {outdir}/t2k_{material}_{cc0pi|cc1pi}{tag}_batch00.npz (QE=channel0, RES=channel1)."""
    os.makedirs(outdir, exist_ok=True)
    prot, _ = _topk(B, lambda pid: pid == 2212, 4)
    neut, _ = _topk(B, lambda pid: pid == 2112, 4)
    counts = {p: _mult_count(B, p) for p in (2212, 2112, 211, 111, -211)}
    out = {}
    for ch, chan in ((0, "cc0pi"), (1, "cc1pi")):
        s = B["channel"] == ch
        d = dict(n_p=counts[2212][s], n_n=counts[2112][s], n_pip=counts[211][s],
                 n_pi0=counts[111][s], n_pim=counts[-211][s],
                 mu=B["k_mu"][s].astype(np.float64), w=B["w0"][s].astype(np.float64),
                 prot=prot[s].astype(np.float64), neut=neut[s].astype(np.float64))
        path = f"{outdir}/t2k_{material}_{chan}{tag}_batch00.npz"
        np.savez(path, **d); out[chan] = (path, int(s.sum()))
    return out


def main():
    bankdir = sys.argv[1] if len(sys.argv) > 1 else "output/event_bank"
    B = BP.load_bank(bankdir)
    qe = "/tmp/bank_schema_qe.npz"; res = "/tmp/bank_schema_res.npz"
    nq = schema(B, 0, qe); nr = schema(B, 1, res)
    print(f"wrote rich-schema: QE(ch0) {nq} evt -> {qe} ; RES(ch1) {nr} evt -> {res}", flush=True)
    env = dict(os.environ, ADONIS_QE_BANK=qe, ADONIS_RES_BANK=res)
    subprocess.run([sys.executable, "-m", "analysis.t2k.make_plots", "--matrix", "--material", "C", "--pdf"],
                   env=env, check=True)


if __name__ == "__main__":
    main()
