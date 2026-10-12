"""Fast, statistically strong, but *not* cryptographically secure generators.

* :class:`SplitMix64` -- tiny pure-Python generator, no dependencies.
* :class:`PCG64`, :class:`Philox`, :class:`SFC64` -- wrappers around numpy's
  implementations (needs numpy: ``pip install 'randomness[numpy]'``).

Do not use these for secrets; see :mod:`randomness.bit_generators.csrng`.
"""

from ._prng import PCG64, SFC64, Philox, SplitMix64

__all__ = ["SplitMix64", "PCG64", "Philox", "SFC64"]
