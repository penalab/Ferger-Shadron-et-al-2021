"""Package for managing ephys data in the Pena Lab

Written with "array data" in mind which is distributed over corresponding files
from both xdphys and the Plexon suite.
"""

# Make sure, we load the extended functionality to pypl2:
from . import extend_pypl2

# Expose the `Site` class as it is basically the one thing users need:
from .site import Site
__all__ = ['Site']
