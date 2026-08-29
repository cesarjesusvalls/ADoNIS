"""Fisher information per subset and per-bin gradient shape, both from one Jacobian npz.

    python -m analysis.paper.grad_info.make [names...] [--label L]
"""
from analysis.paper._driver import run

FIGURES = {
    "fisher":    ("analysis.paper.grad_info.fisher", "main"),
    "gradients": ("analysis.paper.grad_info.gradients", "main"),
}

DEFAULT_LABEL = "multisample_carbon"


if __name__ == "__main__":
    run("analysis.paper.grad_info.make", FIGURES, label=DEFAULT_LABEL)
