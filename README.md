# diffpi

A differentiable Monte-Carlo surrogate for ACHILLES single-pion production and pion
propagation, built so its underlying physics parameters can be tuned against data by
gradient descent. See the design document `../STRATEGY.md` for the full plan.

The core trick (from `../differentiable-sampling/`) is the score-function weight
`score_weight(p) = p / stop_gradient(p)` — unit forward value, gradient `d log p` — which
makes a Monte-Carlo histogram differentiable in the sampling parameters without moving any
bin index. Combined with expected-value deposits (for variance reduction) and detaching the
random-walk geometry, this lets us recover physics parameters from a histogram alone.

## Layout

| file | purpose |
|------|---------|
| `diffpi/kernel.py`        | `score_weight`, parameter bijections, pytree Adam (float64) |
| `diffpi/chain.py`         | compositional driver: one global weight, `RAW`/`CHOICE`/`SHAPE` terms, central geometry detach |
| `diffpi/harness.py`       | validation gates (forward agreement, expected-histogram Jacobian, closure) + gradient-SNR study |
| `diffpi/component_c_r0.py`| Component C rung R0 — 1-D slab, single channel (recover a mean free path) |
| `diffpi/component_c_r1.py`| Component C rung R1 — 3-D sphere, scatter + absorption, tunable angular asymmetry |
| `run_c_r0.py`, `run_c_r1.py` | run the gates + closure and write the figures |

## Run

```bash
pip install -r requirements.txt
python run_c_r0.py      # writes c_r0_closure.png
python run_c_r1.py      # writes c_r1_closure.png
```

## Status

- Component C **R0**: PASS — recovers mean free path L to ~0.004.
- Component C **R1**: PASS — jointly recovers `(L_scatter, L_abs, g)` to <0.02 from one
  exit-angle histogram.

Validation gates live in `harness.py`. The reliable gradient gate is the **expected-histogram
Jacobian** (autodiff vs finite-difference of the smooth `E[hist]`); closure is the definitive
test. See `../STRATEGY.md` §9–§15 for the weight algebra, the toy→realistic ladder, and the
known differentiable-MC gotchas (detach sampled values before the shape score weight; use the
unbiased two-replica MSE; track `renorm = n_data/n_model`).
