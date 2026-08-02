"""Paper section 5: computational performance & novel methodologies enabled by a DIFFERENTIABLE engine.

Exact gradients/Hessians/higher derivatives are ~free, which makes fitting fast (Newton-decrement
quadratic convergence) AND unlocks statistical-interpretability methods a finite-difference generator
cannot afford: non-Gaussian contours from autodiff derivatives, favourable scaling in events and order,
and hybrid grid-bad x Taylor-good schemes.  See docs/paper_plan.md and the sec4_closure engines this reuses."""
