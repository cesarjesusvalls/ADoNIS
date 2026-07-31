"""Section-1 / appendix validation figures: the render HOOKS for the non-histogram observables
(fig1/2/3/456/10/11/13 -- sigma(E), sigma(p), sigma(W), dsigma/domega, angular, sliced double-diff).

These modules expose render(spec) and are dispatched, alongside the pure-config figures, by the ONE
figure entry point analysis/paper/figures/make.py -- every figure is a YAML spec there, rendered the
same way under the same paper style.  Add a figure by dropping a spec in analysis/paper/figures/.
"""
