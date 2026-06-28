"""Plot-time consumer of the differentiable EVENT BANK (event_bank.py).  NO JAX cascade here -- everything
is a cheap re-sum over the stored per-event (kinematics, ragged final state, w0, d^1/d^2/d^3 w/dtheta).

Pick ANY of these at plot time, no re-running:
  * signal definition  -> a boolean mask over events (topology from the full final state + phase space)
  * observable         -> a per-event value (dpt, dat, p_N, muon kinematics, ...)
  * binning            -> any edges
  * forward histogram  (sum w0)            : the T2K-style distribution
  * gradient histogram (sum dw/dtheta_k)   : per-knob Jacobian
  * curvature          : diagonal 2nd/3rd (stored) + off-diagonal Hessian = sum_i Ji*Jj/w0 (reconstructed)

PIDs: proton 2212, neutron 2112, pions {211,111,-211}.
"""
import os, sys, json, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

PION_PIDS = (211, 111, -211)


def load_bank(outdir):
    """Concatenate all chunks into one in-memory bank.  Per-event weights/derivs are divided by n_chunks so
    that summing over ALL events yields the cross section directly (each chunk is an independent estimator)."""
    man = json.load(open(f"{outdir}/manifest.json"))
    nchunks = man["n_chunks"]; files = sorted(glob.glob(f"{outdir}/chunk_*.npz"))
    ev = dict(channel=[], prim_pi_pid=[], w0=[], D1=[], D2=[], D3=[], k_nu=[], p_struck=[], k_mu=[])
    fs_pid = []; fs_chg = []; fs_p4 = []; offs = [np.array([0], np.int64)]
    for f in files:
        d = np.load(f)
        for key in ev:
            ev[key].append(d[key])
        fs_pid.append(d["fs_pid"]); fs_chg.append(d["fs_chg"]); fs_p4.append(d["fs_p4"])
        offs.append(offs[-1][-1] + d["fs_off"][1:])           # rebase each chunk's CSR offsets
    B = {k: np.concatenate(v) for k, v in ev.items()}
    B["w0"] = B["w0"] / nchunks                                 # -> sum = cross section
    for k in ("D1", "D2", "D3"):
        B[k] = B[k] / nchunks
    B["fs_pid"] = np.concatenate(fs_pid); B["fs_chg"] = np.concatenate(fs_chg)
    B["fs_p4"] = np.concatenate(fs_p4); B["fs_off"] = np.concatenate(offs)
    B["labels"] = man["labels"]; B["n_chunks"] = nchunks
    n = len(B["w0"]); B["_eidx"] = np.repeat(np.arange(n), np.diff(B["fs_off"]))
    return B


# ---- per-event reductions over the ragged final state ----------------------------------------------- #
def _event_sum(B, per_particle):
    n = len(B["w0"])
    return np.bincount(B["_eidx"], weights=per_particle, minlength=n)


def n_pions(B):
    return _event_sum(B, np.isin(B["fs_pid"], PION_PIDS).astype(float)).astype(int)


def n_protons(B, pmin=0.0, pmax=np.inf):
    mom = np.linalg.norm(B["fs_p4"][:, 1:], axis=1)
    sel = (B["fs_pid"] == 2212) & (mom >= pmin) & (mom < pmax)
    return _event_sum(B, sel.astype(float)).astype(int)


def leading_proton(B):
    """(lead_p4 (n,4), has_proton (n,)) -- global max-momentum escaped proton per event."""
    n = len(B["w0"]); pid = B["fs_pid"]; p4 = B["fs_p4"]; eidx = B["_eidx"]
    mom = np.linalg.norm(p4[:, 1:], axis=1)
    key = np.where(pid == 2212, mom, -1.0)
    maxk = np.full(n, -1.0); np.maximum.at(maxk, eidx, key)
    lead = np.zeros((n, 4)); islead = (pid == 2212) & (key == maxk[eidx])
    lead[eidx[islead]] = p4[islead]
    return lead, maxk > 0.0


# ---- observables + selections (reuse the validated tune formulas) ------------------------------------ #
def _tune():
    if not hasattr(_tune, "_T"):
        keep = sys.argv; sys.argv = [keep[0], "dpt"]
        from analysis.t2k.differentiability import tune as T
        sys.argv = keep; _tune._T = T
    return _tune._T


def dpt(B, lead):  return np.asarray(_tune()._dpt(B["k_mu"].astype(float), lead))
def dat(B, lead):  return np.asarray(_tune()._dat(B["k_mu"].astype(float), lead))
def acceptance(B, lead):  return np.asarray(_tune()._sel(B["k_mu"].astype(float), lead))


def signal_cc0pi(B, topological=False):
    """model_hist_full CC0pi (primary pion absorbed: prim_pi_pid==0) + acceptance; topological=True instead
    vetoes ANY surviving pion (true 0-meson final state)."""
    lead, has = leading_proton(B)
    sel = acceptance(B, lead) & has
    pi_ok = (n_pions(B) == 0) if topological else (B["prim_pi_pid"] == 0)
    return sel & pi_ok, lead


# ---- histograms (accumulate in f64) ----------------------------------------------------------------- #
def hist_forward(values, B, mask, edges, conv=1.0):
    h, _ = np.histogram(values[mask], bins=edges, weights=B["w0"][mask].astype(np.float64))
    return h / np.diff(edges) * conv


def hist_gradient(values, B, mask, edges, conv=1.0):
    """(nbins, nknob) per-bin gradient d(dsigma/dx)/dtheta from the stored D1."""
    nb = len(edges) - 1; nk = B["D1"].shape[1]; out = np.zeros((nb, nk))
    idx = np.clip(np.searchsorted(edges, values) - 1, 0, nb - 1)
    for k in range(nk):
        out[:, k] = np.bincount(idx[mask], weights=B["D1"][mask, k].astype(np.float64), minlength=nb)
    return out / np.diff(edges)[:, None] * conv


def hist_diag_curv(values, B, mask, edges, order, conv=1.0):
    """Per-bin diagonal 2nd (order=2) or 3rd (order=3) derivative histogram from stored D2/D3."""
    D = B["D2"] if order == 2 else B["D3"]
    nb = len(edges) - 1; nk = D.shape[1]; out = np.zeros((nb, nk))
    idx = np.clip(np.searchsorted(edges, values) - 1, 0, nb - 1)
    for k in range(nk):
        out[:, k] = np.bincount(idx[mask], weights=D[mask, k].astype(np.float64), minlength=nb)
    return out / np.diff(edges)[:, None] * conv


if __name__ == "__main__":
    B = load_bank(sys.argv[1] if len(sys.argv) > 1 else "output/event_bank")
    print(f"loaded {len(B['w0'])} events ({(B['channel']==0).sum()} QE + {(B['channel']==1).sum()} RES), "
          f"{len(B['fs_pid'])} final-state particles, {B['n_chunks']} chunk(s)", flush=True)
