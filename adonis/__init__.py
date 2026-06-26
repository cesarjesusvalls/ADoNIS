"""ADoNIS -- A Differentiable generatOr of Neutrino Interaction Samples.

A modular, differentiable Monte-Carlo surrogate for ACHILLES neutrino interactions. The
chain is built from swappable components (flux, nuclear model, primary channel, FSI) around
the kind-1 sample/reweight contract, with tunable physics parameters centralised in
`PhysicsParams`.

Top-level convenience API:
    from adonis import PhysicsParams, ChainConfig, Generator, DCCSinglePion, NoFSI
    from adonis import SpectralFunction, Monochromatic
    from adonis import observables, fit
"""
__version__ = "0.1.0"

from adonis.core.params import PhysicsParams, ChainConfig, DCCKnobs          # noqa: F401
from adonis.core.chain import Generator                              # noqa: F401
from adonis.primary.dcc.channel import DCCSinglePion                 # noqa: F401
from adonis.fsi.none import NoFSI                                    # noqa: F401
from adonis.nuclear.spectral import SpectralFunction                 # noqa: F401
from adonis.nuclear.free import FreeNucleon                          # noqa: F401
from adonis.flux.mono import Monochromatic                           # noqa: F401
from adonis import observables                                       # noqa: F401
from adonis.analysis import fit                                      # noqa: F401
