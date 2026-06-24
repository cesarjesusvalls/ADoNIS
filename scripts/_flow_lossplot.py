"""On-request LIVE training-status plot from a flow loss-history npz (written/flushed by the trainer
every few steps).  Loss + N_eff/N + gradient-norm vs step.  Safe to run mid-training.
Usage: python scripts/_flow_lossplot.py [/tmp/flow_toy_loss_s0.npz] [out.png]"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/flow_toy_loss_s0.npz"
out = sys.argv[2] if len(sys.argv) > 2 else "paper_figures/flow_training_status.png"
d = np.load(src)
s = d["step"]; loss = d["loss"]; neff = d["neff"] if "neff" in d else d["neffN"]
has_g = "gnorm" in d
n = 2 + int(has_g)
fig, ax = plt.subplots(1, n, figsize=(5.0 * n, 4))
ax[0].plot(s, loss, color="crimson"); ax[0].set_yscale("log")
ax[0].set_xlabel("step"); ax[0].set_ylabel("loss = E[w^2]/E[w]^2"); ax[0].set_title("training loss")
ax[1].plot(s, neff, color="navy"); ax[1].axhline(0.32, ls=":", c="gray", label="Vegas 0.32")
ax[1].axhline(0.092, ls=":", c="orange", label="flat/separable 0.092")
ax[1].set_xlabel("step"); ax[1].set_ylabel("N_eff/N (fresh sample)")
ax[1].set_title(f"efficiency (latest {neff[-1]:.3f} @ step {int(s[-1])})"); ax[1].legend(fontsize=8)
if has_g:
    ax[2].plot(s, d["gnorm"], color="seagreen"); ax[2].set_yscale("log")
    ax[2].set_xlabel("step"); ax[2].set_ylabel("grad norm (pre-clip)"); ax[2].set_title("gradient norm")
plt.tight_layout(); plt.savefig(out, dpi=120)
print(f"wrote {out}  (step {int(s[-1])}, N_eff/N {neff[-1]:.3f}, max {neff.max():.3f})")
