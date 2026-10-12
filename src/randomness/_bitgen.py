"""The :class:`BitGenerator` protocol: raw 64-bit randomness for :class:`Generator`."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["BitGenerator"]

MASK64 = (1 << 64) - 1


@runtime_checkable
class BitGenerator(Protocol):
    """Anything that can produce uniformly distributed 64-bit unsigned ints.

    Only ``random_raw`` is required.  Three optional members add features to
    :class:`~randomness.Generator`:

    * ``seed(*args, **kwargs)`` enables ``Generator.seed``.
    * a ``state`` property (readable and writable) enables ``getstate``,
      ``setstate``, ``copy`` and ``pickle``.
    * ``random_bytes(n)`` speeds up ``Generator.randbytes``.  It must return
      the little-endian bytes of the next ``ceil(n / 8)`` ``random_raw()``
      words, cut to ``n`` bytes.

    numpy's 64-bit bit generators (``PCG64``, ``Philox``, ``SFC64``) work
    as-is, minus ``Generator.seed`` (they have no ``seed`` method; build a
    new one).

    The values must really span 64 bits: ``Generator`` uses the high bits, so
    a 32-bit source (e.g. numpy's ``MT19937.random_raw``) yields ``random()``
    values near 0 and can make rejection loops such as ``normalvariate`` spin
    forever.
    """

    def random_raw(self) -> int:
        """Advance the state and return the next uniform 64-bit unsigned int."""
        ...
