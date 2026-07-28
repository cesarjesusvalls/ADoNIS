"""ADoNIS -- A Differentiable generatOr of Neutrino Interaction Samples.

A modular, differentiable Monte-Carlo surrogate for ACHILLES neutrino interactions. The
chain is built from swappable components (flux, nuclear model, primary channel, FSI) around
the kind-1 sample/reweight contract, with tunable physics parameters centralised in
`PhysicsParams`.

Top-level convenience API:
    from adonis import PhysicsParams, ChainConfig, Generator, DCCSinglePion, NoFSI
    from adonis import SpectralFunction, Monochromatic
    from adonis import kinematics, fit
"""
__version__ = "0.1.0"

# LAZY re-exports (PEP 562).  Eagerly importing the generator stack here pulled JAX into EVERY consumer of
# the package -- so `import adonis.workflow.plotting` (pure numpy+matplotlib) paid a ~2s JAX import it never
# uses.  The convenience API below still works (`from adonis import Generator` imports the submodule ON FIRST
# USE); lightweight consumers (plotting, config, analysis I/O) no longer pull JAX at all.  Submodule access
# (`from adonis import kinematics`, `from adonis.analysis import fit`) works via the normal import machinery.
_LAZY = {
    "PhysicsParams": "adonis.core.params", "ChainConfig": "adonis.core.params", "DCCKnobs": "adonis.core.params",
    "Generator": "adonis.core.chain", "DCCSinglePion": "adonis.channels.dcc.channel",
    "NoFSI": "adonis.fsi.none", "SpectralFunction": "adonis.nuclear.spectral",
    "FreeNucleon": "adonis.nuclear.free", "Monochromatic": "adonis.flux.mono",
}


def __getattr__(name):                       # PEP 562 module-level lazy attribute
    import importlib
    target = _LAZY.get(name)
    if target is not None:
        return getattr(importlib.import_module(target), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
