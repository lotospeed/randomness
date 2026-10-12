"""Implementation of :mod:`randomness.bit_generators.prng`."""

from __future__ import annotations

import os
import struct
from typing import Any

from .._bitgen import MASK64

__all__ = ["SplitMix64", "PCG64", "Philox", "SFC64"]


class SplitMix64:
    """SplitMix64 (Steele, Lea & Flood 2014): a tiny, fast 64-bit generator.

    The whole state is a single 64-bit integer, so it is trivial to seed,
    save and restore.  Good for general use and for seeding other generators;
    use a larger-state generator if you need many parallel streams.
    """

    __slots__ = ("_state",)

    def __init__(self, seed: int | None = None) -> None:
        self._state = 0
        self.seed(seed)

    def seed(self, a: int | None = None) -> None:
        """Seed from an int (any size; reduced mod 2**64) or, for ``None``, OS entropy."""
        if a is None:
            a = int.from_bytes(os.urandom(8), "little")
        elif not isinstance(a, int):
            raise TypeError(f"seed must be an int or None, not {type(a).__name__}")
        self._state = a & MASK64

    @property
    def state(self) -> int:
        return self._state

    @state.setter
    def state(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"state must be an int, not {type(value).__name__}")
        self._state = value & MASK64

    def random_raw(self) -> int:
        self._state = s = (self._state + 0x9E3779B97F4A7C15) & MASK64
        s = ((s ^ (s >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        s = ((s ^ (s >> 27)) * 0x94D049BB133111EB) & MASK64
        return s ^ (s >> 31)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(state={self._state:#018x})"

    def __getstate__(self) -> int:
        return self._state

    def __setstate__(self, state: Any) -> None:
        self._state = state


def _numpy_random() -> Any:
    try:
        import numpy.random as np_random
    except ImportError as e:
        raise ImportError(
            "this bit generator wraps numpy: pip install 'randomness[numpy]'"
        ) from e
    return np_random


class _NumpyBitGenerator:
    """Wraps a numpy bit generator, adding seed, state and buffering.

    The numpy implementation produces the values: the stream is exactly
    numpy.random.<Name>(seed).random_raw(n).  Words are pulled from numpy
    in batches of _REFILL_WORDS, which is much faster than one call per
    word.  state is {'bit_generator': <numpy state dict>, 'buffer':
    <unread words>}; the numpy state alone is ahead of what has been served.

    seed is anything numpy accepts (int, SeedSequence, array of ints)
    or None for OS entropy.  Advanced numpy-only features such as
    jumped or Philox counter/key are not exposed.
    """

    __slots__ = ("_bg", "_words")

    _NUMPY_NAME: str
    _REFILL_WORDS = 1024

    def __init__(self, seed: Any = None) -> None:
        self._bg = self._make(seed)
        self._words: list[int] = []

    @classmethod
    def _make(cls, seed: Any) -> Any:
        return getattr(_numpy_random(), cls._NUMPY_NAME)(seed)

    def seed(self, a: Any = None) -> None:
        """Replace the numpy generator with a freshly seeded one."""
        self._bg = self._make(a)
        self._words = []

    def random_raw(self) -> int:
        """Return the next 64-bit unsigned int."""
        words = self._words
        if not words:
            words = self._bg.random_raw(self._REFILL_WORDS).tolist()
            words.reverse()  # so pop() yields them in output order
            self._words = words
        return words.pop()

    @property
    def state(self) -> dict[str, Any]:
        n = len(self._words)
        return {
            "bit_generator": self._bg.state,
            "buffer": struct.pack(f"<{n}Q", *reversed(self._words)),
        }

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        buffer = value["buffer"]
        if (
            not isinstance(buffer, bytes)
            or len(buffer) % 8
            or len(buffer) > 8 * self._REFILL_WORDS
        ):
            raise ValueError(
                "state['buffer'] must be bytes, a multiple of 8 long, "
                f"at most {8 * self._REFILL_WORDS}"
            )
        self._bg.state = value["bit_generator"]  # numpy validates this
        self._words = list(reversed(struct.unpack(f"<{len(buffer) // 8}Q", buffer)))

    def __getstate__(self) -> dict[str, Any]:
        return self.state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._bg = self._make(0)
        self.state = state

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class PCG64(_NumpyBitGenerator):
    """numpy's PCG64 (O'Neill's PCG XSL-RR 128/64)."""

    __slots__ = ()
    _NUMPY_NAME = "PCG64"


class Philox(_NumpyBitGenerator):
    """numpy's Philox (4x64, 10 rounds; counter-based)."""

    __slots__ = ()
    _NUMPY_NAME = "Philox"


class SFC64(_NumpyBitGenerator):
    """numpy's SFC64 (Small Fast Chaotic, 256-bit state)."""

    __slots__ = ()
    _NUMPY_NAME = "SFC64"


# The public home of these classes (and their pickle path) is the prng module.
for _cls in (SplitMix64, PCG64, Philox, SFC64):
    _cls.__module__ = "randomness.bit_generators.prng"
del _cls
