"""Compare ACHILLES hadron current zj_mu (ZJDUMP from amplitude()) vs my exclusive gather,
via the spin-summed hadron tensor W^{mu,nu}=sum_spin conj(zj_mu) zj_nu (basis-independent,
so the spin rotations I skip don't matter). Reveals WHICH mu,nu (longitudinal 0,3 vs
transverse 1,2) carries the irot_q=1 deficit. Global fac/_NORM cancels in the ratio pattern."""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
from adonis.xsec.dcc_current import exclusive_amps2_batch

LOG = Path("/Users/homelab/Lab/Playground/projects/DIFFGEN/Achilles/_zmtxout/zjrun.log")


def parse():
    blocks = []
    cur = None
    with open(LOG) as f:
        for line in f:
            if line.startswith("ZJDUMP"):
                if cur: blocks.append(cur)
                cur = {"zj": np.zeros((2, 2, 4), complex)}
            elif cur is None:
                continue
            elif line.startswith("ZJ_qvec"):
                cur["q"] = np.array([float(x) for x in line.split()[1:5]])
            elif line.startswith("ZJ_xpnuc"):
                cur["pN"] = np.array([float(x) for x in line.split()[1:5]])
            elif line.startswith("ZJ_xk"):
                cur["pi"] = np.array([float(x) for x in line.split()[1:5]])
            elif line.startswith("ZJVAL"):
                m = re.search(r"is1=\s*(-?\d+)\s+is2=\s*(-?\d+)\s*:\s*(.*)", line)
                i1 = (int(m.group(1)) + 1) // 2; i2 = (int(m.group(2)) + 1) // 2
                v = [float(x) for x in m.group(3).split()]
                cur["zj"][i1, i2] = [v[2*k] + 1j*v[2*k+1] for k in range(4)]
    if cur: blocks.append(cur)
    return blocks


def Wtensor(zj_combo):
    """zj (..., 4mu) over spin combos -> W[mu,nu]=sum_combo conj(zj_mu) zj_nu."""
    z = zj_combo.reshape(-1, 4)
    return np.einsum('cm,cn->mn', np.conj(z), z)


def main():
    blocks = parse()
    print(f"parsed {len(blocks)} ZJDUMP blocks\n")
    dummy = np.zeros((1, 4))
    for b in blocks:
        q, pN, pi = b["q"], b["pN"], b["pi"]
        Wa = Wtensor(b["zj"])                       # ACHILLES (has global fac^2)
        zj_mine = np.asarray(exclusive_amps2_batch(
            dummy, dummy, dummy.copy() + [938.272, 0, 0, 0],
            pN[None], pi[None], +1, 211, return_zj=True, q_direct=q[None]))[0]   # (4combo,4mu)
        Wm = Wtensor(zj_mine)
        Q2 = (q[1:] @ q[1:] - q[0]**2) / 1e6
        pcm = pN + pi; W = np.sqrt(max(pcm[0]**2 - pcm[1:] @ pcm[1:], 0))
        # global scale: match on transverse trace (W11+W22), then compare every component
        sa = (Wa[1, 1] + Wa[2, 2]).real; sm = (Wm[1, 1] + Wm[2, 2]).real
        scale = sa / sm if sm != 0 else 0.0
        print(f"=== W={W:.0f} MeV  Q2={Q2:.3f} GeV^2   (transverse-matched scale={scale:.3e}) ===")
        diag = ["W00(time)", "W11(trans)", "W22(trans)", "W33(z)"]
        for mu in range(4):
            a = Wa[mu, mu].real; m = Wm[mu, mu].real * scale
            r = m / a if abs(a) > 1e-30 else float('nan')
            tag = "  <== LONGITUDINAL" if mu in (0, 3) else ""
            print(f"  {diag[mu]:11s}: ACH={a:+.4e}  MINE*s={m:+.4e}  ratio={r:.3f}{tag}")
        # off-diagonal components incl. the transverse-longitudinal interference (W01,W13,...)
        for (a_, b_, nm) in [(0,3,"W03 time-z"), (0,1,"W01 time-x"), (0,2,"W02 time-y"),
                             (1,3,"W13 z-x"), (2,3,"W23 z-y"), (1,2,"W12 x-y")]:
            av = Wa[a_, b_]; mv = Wm[a_, b_] * scale
            rr = (mv.real / av.real) if abs(av.real) > 1e-6 * abs(Wa[1,1]) else float('nan')
            tag = "  <== T-L interference" if (a_ in (0,3) and b_ in (1,2)) else ""
            print(f"  {nm:11s}: ACH={av.real:+.3e}{av.imag:+.3e}j  MINE*s={mv.real:+.3e}{mv.imag:+.3e}j  ratio(re)={rr:.3f}{tag}")
        print()


if __name__ == "__main__":
    main()
