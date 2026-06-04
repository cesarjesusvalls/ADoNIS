"""High-statistics, same-binning comparison of the faithful full-port fold to the
neutrino oracle, to resolve the residual low-Q^2 peak.

Oracle: the original 1M-event histograms (oracle_distributions_nu.npz, 30 native bins
-> low per-bin noise). Model: the faithful fold (dcc_fold_full) run in CHUNKS and
accumulated to many millions of events, histogrammed on the SAME oracle bin edges.

Run:  python compare_highstats.py
"""
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffpi.dcc import DCCKnobs
from diffpi.hadron_xsec import HadronStructure
from diffpi.dcc_fold_full import fold_full_events

jax.config.update("jax_enable_x64", True)

o = np.load("oracle/oracle_distributions_nu.npz")
We, dW_or = o["W_edges"], o["dsig_W"]
Q2e, dQ_or = o["Q2_edges"], o["dsig_Q2"]
Wc = 0.5 * (We[:-1] + We[1:]); Q2c = 0.5 * (Q2e[:-1] + Q2e[1:])

hs = HadronStructure(n_theta=16, n_phi=16)            # finer angular quadrature too
N_CHUNK, N_CHUNKS = 1_000_000, 12
hW = np.zeros(len(Wc)); hQ = np.zeros(len(Q2c))
for c in range(N_CHUNKS):
    W, Q2, w = fold_full_events(DCCKnobs(), jax.random.PRNGKey(1000 + c), n=N_CHUNK, hs=hs)
    W, Q2, w = np.asarray(W), np.asarray(Q2), np.asarray(w)
    hW += np.histogram(W, bins=We, weights=w)[0]
    hQ += np.histogram(Q2, bins=Q2e, weights=w)[0]
    print(f"  chunk {c+1}/{N_CHUNKS} done", end="\r")
print(f"\nmodel total events: {N_CHUNK*N_CHUNKS:,}")

hW /= np.diff(We); hQ /= np.diff(Q2e)


def unit(h, c):
    a = np.trapz(h, c); return h / a if a > 0 else h


hWm, hWo = unit(hW, Wc), unit(dW_or, Wc)
hQm, hQo = unit(hQ, Q2c), unit(dQ_or, Q2c)
rW = float(np.sum((hWm - hWo) ** 2) / np.sum(hWo ** 2))
rQ = float(np.sum((hQm - hQo) ** 2) / np.sum(hQo ** 2))
print(f"dsigma/dW  relL2 = {rW:.5f}")
print(f"dsigma/dQ2 relL2 = {rQ:.5f}")
# per-bin ratio in the Q^2 peak region
print("Q^2 peak bins (model/oracle):")
for i in range(min(6, len(Q2c))):
    print(f"  Q2={Q2c[i]/1e6:.3f} GeV^2: model/oracle = {hQm[i]/hQo[i]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(Wc, hWo, "o-", ms=4, color="C1", label="oracle (1M)")
ax[0].plot(Wc, hWm, "-", lw=2, color="C3", label=f"fold ({N_CHUNK*N_CHUNKS//1_000_000}M)")
ax[0].set_xlabel("W [MeV]"); ax[0].set_ylabel("norm d$\\sigma$/dW")
ax[0].set_title(f"d$\\sigma$/dW  (relL2={rW:.1e})"); ax[0].legend()
ax[1].plot(Q2c / 1e6, hQo * 1e6, "o-", ms=4, color="C1", label="oracle (1M)")
ax[1].plot(Q2c / 1e6, hQm * 1e6, "-", lw=2, color="C3", label=f"fold ({N_CHUNK*N_CHUNKS//1_000_000}M)")
ax[1].set_xlabel(r"$Q^2$ [GeV$^2$]"); ax[1].set_ylabel("norm d$\\sigma$/d$Q^2$")
ax[1].set_title(f"d$\\sigma$/d$Q^2$  (relL2={rQ:.1e})"); ax[1].legend()
fig.suptitle("Faithful fold vs neutrino oracle -- high statistics, native bins")
fig.tight_layout(); fig.savefig("fold_full_highstats.png", dpi=120)
print("saved -> fold_full_highstats.png")
