"""Bit generators for :class:`randomness.Generator`.

Two sections:

* :mod:`randomness.bit_generators.prng` -- fast, statistically strong,
  *not* cryptographically secure.
* :mod:`randomness.bit_generators.csrng` -- cryptographically secure.

Import the generators from their section.
"""

from . import csrng, prng

__all__ = ["csrng", "prng"]
