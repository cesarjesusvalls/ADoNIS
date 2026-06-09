"""Component-by-component zmtx comparison: ACHILLES interpolate_amp dump vs my build_zmtx,
at the EXACT (W,Q2,itiz) ACHILLES used.  Pinpoints the longitudinal-sector divergence.

Parse /Achilles/_zmtxout/run_full.log ZMTXDUMP/ZMTXVAL blocks; for each (W,Q2) compute my
zmtx via the spline interpolation + build_zmtx; print ACH vs MINE for every (pw, idx)."""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from adonis.primary.dcc import amplitudes as AMP
from adonis.primary.dcc.assembly import build_zmtx
from adonis.primary.dcc.amplitudes import DCCKnobs
from adonis.primary.dcc.loader import load_cached
import adonis.xsec.constants as C

LOG = Path("/Users/homelab/Lab/Playground/projects/DIFFGEN/Achilles/_zmtxout/run_full.log")
_amp = AMP.DCCAmplitudes()
_T = load_cached()
_2J, _2L, _2I = _T.pw_2J, _T.pw_2L, _T.pw_2I


def parse_dumps():
    blocks = []
    cur = None
    with open(LOG) as f:
        for line in f:
            if line.startswith("ZMTXDUMP"):
                if cur is not None:
                    blocks.append(cur)
                m = dict(re.findall(r"(\w+)=\s*([-+0-9.eE]+)", line))
                cur = {"W": float(m["wcmMeV"]), "Q2": float(m["Q2MeV2"]),
                       "itiz": int(float(m["itiz"])), "vals": {}}
            elif line.startswith("ZMTXVAL") and cur is not None:
                m = dict(re.findall(r"(\w+)=\s*([-+0-9.eE]+)", line))
                pw = int(float(m["pw"])) - 1; idx = int(float(m["idx"])) - 1
                cur["vals"][(pw, idx)] = float(m["re"]) + 1j * float(m["im"])
    if cur is not None:
        blocks.append(cur)
    return blocks


def my_zmtx(W, Q2, itiz, mpi):
    knobs = DCCKnobs()
    # spline interpolation (matches ACHILLES interpolate_amp) at single (W,Q2)
    vec, isv, axial = _amp.amplitudes_spline_np(np.array([W]), np.array([Q2]), knobs)
    z = build_zmtx(vec[0], isv[0], axial[0], jnp.asarray(W), jnp.asarray(Q2),
                   _2J, _2L, _2I, mode=1, itiz=itiz, m_N=C.mN, m_pi=mpi)
    return np.asarray(z)            # (8, npw)


def main():
    mpi = float(sys.argv[1]) if len(sys.argv) > 1 else 138.039   # match ACHILLES fpio
    blocks = parse_dumps()
    print(f"parsed {len(blocks)} (W,Q2) blocks; m_pi={mpi}  C.mN={C.mN:.3f}\n")
    # idx of interest: longitudinal time(2,3) and z(6,7) 0-based; transverse 0,1,4,5
    LONG = [2, 3, 6, 7]
    for b in blocks[:4]:
        z = my_zmtx(b["W"], b["Q2"], b["itiz"], mpi)
        npw = z.shape[1]
        print(f"=== W={b['W']:.1f} Q2={b['Q2']:.0f} (GeV2={b['Q2']*1e-6:.3f}) itiz={b['itiz']} ===")
        for pw in range(npw):
            # only print pw with appreciable ACH content
            ach_mag = sum(abs(b["vals"].get((pw, k), 0)) for k in range(8))
            if ach_mag < 1e-9:
                continue
            print(f" pw{pw} 2J={_2J[pw]} 2L={_2L[pw]} 2I={_2I[pw]}:")
            for k in range(8):
                a = b["vals"].get((pw, k), 0 + 0j); m = z[k, pw]
                tag = "  <==LONG" if k in LONG else ""
                rr = (abs(m) / abs(a)) if abs(a) > 1e-12 else float('nan')
                print(f"   idx{k+1}: ACH={a.real:+.4e}{a.imag:+.4e}j  "
                      f"MINE={m.real:+.4e}{m.imag:+.4e}j  |M|/|A|={rr:.3f}{tag}")
        print()


if __name__ == "__main__":
    main()
