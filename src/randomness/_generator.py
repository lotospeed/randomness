"""A ``random.Random`` driven entirely by a :class:`BitGenerator`."""

from __future__ import annotations

import copy
import operator
import random
import struct
from typing import Any

from ._bitgen import MASK64, BitGenerator

__all__ = ["Generator"]


class Generator(random.Random):
    """The full :class:`random.Random` API, fed by a ``BitGenerator``.

    Like :class:`numpy.random.Generator`, this class owns no entropy of its
    own: every value it returns is derived from calls to
    ``bit_generator.random_raw()``.  ``random.Random`` already implements
    every distribution in terms of ``random()`` and ``getrandbits()``, so those
    two are all that is defined here; ``randrange``, ``shuffle``, ``sample``,
    ``gauss`` etc. are inherited unchanged.

    Bit usage is deliberately stateless: ``random()`` and every
    ``getrandbits(k)`` with ``k <= 64`` consume exactly one raw word (the high
    bits are used), and larger ``k`` consume ``ceil(k / 64)`` words.  The only
    state beyond the bit generator's is the cached ``gauss_next`` value.

    ``randbytes`` is also overridden, to take bytes straight from bit
    generators that offer ``random_bytes``; its output is unchanged.

    Note that the hidden Mersenne Twister of the ``random.Random`` base class
    is never touched.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Generator:
        # _random.Random.__new__ would otherwise try to seed its Mersenne
        # Twister from our arguments (hashing the bit generator).
        return super().__new__(cls)

    def __init__(self, bit_generator: BitGenerator) -> None:
        # Deliberately not calling Random.__init__, which would seed().
        if not callable(getattr(bit_generator, "random_raw", None)):
            raise TypeError(
                "bit_generator must provide a random_raw() method returning "
                f"64-bit unsigned ints, got {type(bit_generator).__name__}"
            )
        self._bit_generator = bit_generator
        self._raw = bit_generator.random_raw
        self._raw_bytes = getattr(bit_generator, "random_bytes", None)
        self.gauss_next = None

    @property
    def bit_generator(self) -> BitGenerator:
        """The underlying bit generator."""
        return self._bit_generator

    # -- the two primitives everything else is built from -----------------

    def random(self) -> float:
        """Return the next float in [0.0, 1.0) with 53 bits of precision."""
        return ((self._raw() & MASK64) >> 11) * (1.0 / (1 << 53))

    def getrandbits(self, k: int) -> int:
        """Return a non-negative int with ``k`` random bits."""
        if k < 0:
            raise ValueError("number of bits must be non-negative")
        if k <= 64:
            if k == 0:
                return 0
            return (self._raw() & MASK64) >> (64 - k)
        raw = self._raw
        words, rem = divmod(k, 64)
        # Packed in one go: OR-ing words into a growing int is quadratic.
        result = int.from_bytes(
            struct.pack(f"<{words}Q", *[raw() & MASK64 for _ in range(words)]), "little"
        )
        if rem:
            result |= ((raw() & MASK64) >> (64 - rem)) << (64 * words)
        return result

    def randbytes(self, n: int) -> bytes:
        """Generate ``n`` random bytes.

        Uses the bit generator's ``random_bytes`` when it has one.  The result
        is the same as without it, only faster.
        """
        raw_bytes = self._raw_bytes
        if raw_bytes is None:
            return super().randbytes(n)
        n = operator.index(n)
        if n < 0:
            raise ValueError("number of bytes must be non-negative")
        words, rem = divmod(n, 8)
        if not rem:
            return raw_bytes(n)
        # getrandbits() uses the high bits of a partial word: its last bytes.
        data = raw_bytes(8 * (words + 1))
        return data[: 8 * words] + data[-rem:]

    # -- state management --------------------------------------------------

    def seed(self, *args: Any, **kwargs: Any) -> None:
        """Reseed the bit generator via its own ``seed`` method.

        Raises ``NotImplementedError`` if the bit generator has none.
        """
        seed = getattr(self._bit_generator, "seed", None)
        if seed is None:
            raise NotImplementedError(
                f"{type(self._bit_generator).__name__} does not support seed()"
            )
        seed(*args, **kwargs)
        self.gauss_next = None

    def getstate(self) -> tuple[int, Any, float | None]:
        """Return ``(version, bit_generator.state, gauss_next)``."""
        try:
            state = self._bit_generator.state  # type: ignore[attr-defined]
        except AttributeError:
            raise NotImplementedError(
                f"{type(self._bit_generator).__name__} does not expose a "
                "'state' property"
            ) from None
        return self.VERSION, copy.deepcopy(state), self.gauss_next

    def setstate(self, state: tuple[int, Any, float | None]) -> None:
        version, bitgen_state, gauss_next = state
        if version != self.VERSION:
            raise ValueError(
                f"state with version {version} passed to "
                f"Generator.setstate() of version {self.VERSION}"
            )
        if not hasattr(self._bit_generator, "state"):
            raise NotImplementedError(
                f"{type(self._bit_generator).__name__} does not expose a "
                "'state' property"
            )
        self._bit_generator.state = copy.deepcopy(bitgen_state)  # type: ignore[attr-defined]
        self.gauss_next = gauss_next

    # -- copy / pickle: the bit generator travels with the Generator -------

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (self._bit_generator,), self.gauss_next

    def __setstate__(self, gauss_next: float | None) -> None:
        self.gauss_next = gauss_next

    def __copy__(self) -> Generator:
        new = type(self)(copy.deepcopy(self._bit_generator))
        new.gauss_next = self.gauss_next
        return new

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._bit_generator!r})"
