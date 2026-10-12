"""Cryptographically secure bit generators.

* :class:`SystemBitGenerator` -- the operating system's CSPRNG (``os.urandom``).
* :class:`HMAC_DRBG` -- NIST SP 800-90A HMAC_DRBG (standard library only).
* :class:`ChaCha20` -- ChaCha20 with fast key erasure; needs the optional
  ``pycryptodome`` dependency (``pip install 'randomness[crypto]'``).

The seeded generators are deterministic: they are only as secret as their
seed, and their ``state`` / ``getstate`` / ``copy`` / ``pickle`` output is
key material.  Like any userspace PRNG they also duplicate their state across
``os.fork``; reseed in the child if that matters.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
import weakref
from typing import Any

_chacha20: Any = None  # Crypto.Cipher.ChaCha20, imported on first use

__all__ = ["ChaCha20", "HMAC_DRBG", "SystemBitGenerator"]

_KEY_BYTES = 32
_MIN_RESEED_BYTES = 16
_ZERO_NONCE = bytes(8)


class ChaCha20:
    """ChaCha20 CSPRNG with fast key erasure (Bernstein, 2017).

    Each refill runs ChaCha20 under the current key (zero nonce, counter 0)
    for ``_REFILL_BYTES`` of keystream.  The first 32 bytes become the *next*
    key; the rest are served as little-endian 64-bit words.  Since every key
    is used exactly once, the fixed nonce is safe, and a leaked state reveals
    nothing about words already returned.

    Python cannot reliably zero memory, so the erasure is logical: the old key
    is dropped, not scrubbed.

    Seeds are exactly 32 bytes (or ``None`` for OS entropy).  ``state`` holds
    key material: treat ``getstate()``, ``copy`` and ``pickle`` output as
    secret.
    """

    __slots__ = ("_key", "_words")

    # 16 ChaCha blocks per refill: 32 key bytes + 124 words.
    _REFILL_BYTES = 1024
    _WORDS_PER_REFILL = (_REFILL_BYTES - _KEY_BYTES) // 8

    def __init__(self, seed: bytes | None = None) -> None:
        _require_pycryptodome()
        self._key = bytes(_KEY_BYTES)
        self._words: list[int] = []
        self.seed(seed)

    def seed(self, a: bytes | bytearray | memoryview | None = None) -> None:
        """Set the key from exactly 32 bytes, or from OS entropy if ``None``."""
        if a is None:
            key = os.urandom(_KEY_BYTES)
        elif isinstance(a, (bytes, bytearray, memoryview)):
            key = bytes(a)
            if len(key) != _KEY_BYTES:
                raise ValueError(f"seed must be exactly {_KEY_BYTES} bytes, got {len(key)}")
        else:
            raise TypeError(f"seed must be bytes or None, not {type(a).__name__}")
        self._key = key
        self._words = []

    def reseed(self, entropy: bytes) -> None:
        """Mix ``entropy`` (at least 16 bytes) into the key, keeping existing entropy.

        Buffered words are discarded.
        """
        entropy = bytes(entropy)
        if len(entropy) < _MIN_RESEED_BYTES:
            raise ValueError(f"need at least {_MIN_RESEED_BYTES} bytes of entropy")
        self._key = hashlib.blake2b(entropy, key=self._key, digest_size=_KEY_BYTES).digest()
        self._words = []

    def _refill(self) -> None:
        stream = _chacha20.new(key=self._key, nonce=_ZERO_NONCE).encrypt(
            bytes(self._REFILL_BYTES)
        )
        self._key = stream[:_KEY_BYTES]
        # Reversed so random_raw can pop() from the end in output order.
        self._words = list(
            reversed(struct.unpack(f"<{self._WORDS_PER_REFILL}Q", stream[_KEY_BYTES:]))
        )

    def random_raw(self) -> int:
        """Return the next 64-bit unsigned int."""
        words = self._words
        if not words:
            self._refill()
            words = self._words
        return words.pop()

    @property
    def state(self) -> dict[str, bytes]:
        """``{'key': 32 bytes, 'buffer': unread output bytes}``; secret."""
        n = len(self._words)
        return {
            "key": self._key,
            "buffer": struct.pack(f"<{n}Q", *reversed(self._words)),
        }

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        key, buffer = value["key"], value["buffer"]
        if not isinstance(key, bytes) or len(key) != _KEY_BYTES:
            raise ValueError(f"state['key'] must be {_KEY_BYTES} bytes")
        if (
            not isinstance(buffer, bytes)
            or len(buffer) % 8
            or len(buffer) > 8 * self._WORDS_PER_REFILL
        ):
            raise ValueError("state['buffer'] must be bytes, a multiple of 8 long, "
                             f"at most {8 * self._WORDS_PER_REFILL}")
        self._key = key
        self._words = list(reversed(struct.unpack(f"<{len(buffer) // 8}Q", buffer)))

    def __getstate__(self) -> dict[str, bytes]:
        return self.state

    def __setstate__(self, state: dict[str, bytes]) -> None:
        _require_pycryptodome()
        self.state = state

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<secret state>)"


def _require_pycryptodome() -> None:
    global _chacha20
    if _chacha20 is None:
        try:
            from Crypto.Cipher import ChaCha20 as _cc
        except ImportError as e:
            raise ImportError(
                "ChaCha20 requires pycryptodome: pip install 'randomness[crypto]'"
            ) from e
        _chacha20 = _cc


class HMAC_DRBG:
    """HMAC_DRBG as specified in NIST SP 800-90A rev. 1, section 10.1.2.

    Parameters
    ----------
    seed:
        Entropy input followed by nonce, as one byte string of at least 48
        bytes (32 bytes of entropy + 16 bytes of nonce, the minimum for the
        256-bit security strength of every supported hash).  ``None`` draws 48
        bytes from ``os.urandom``.
    hash:
        ``'sha256'`` (default), ``'sha384'`` or ``'sha512'``.
    personalization:
        Optional personalization string, applied whenever the generator is
        (re)seeded.

    Output is produced by one Generate call per refill of ``_REFILL_BYTES``
    (the standard allows up to 65536 per call), so the backtracking-resistance
    update happens once per buffer, not once per word.
    """

    __slots__ = ("_hash", "_personalization", "_key", "_v", "_counter", "_words")

    _STRENGTH_BYTES = 32
    _MIN_SEED_BYTES = _STRENGTH_BYTES + _STRENGTH_BYTES // 2
    _MAX_REQUEST_BYTES = 65536
    _RESEED_INTERVAL = 1 << 48
    _REFILL_BYTES = 1024
    _DIGEST_SIZES = {"sha256": 32, "sha384": 48, "sha512": 64}

    def __init__(
        self,
        seed: bytes | None = None,
        *,
        hash: str = "sha256",
        personalization: bytes = b"",
    ) -> None:
        if hash not in self._DIGEST_SIZES:
            raise ValueError(f"hash must be one of {sorted(self._DIGEST_SIZES)}")
        self._hash = hash
        self._personalization = bytes(personalization)
        self._words: list[int] = []
        self.seed(seed)

    # -- SP 800-90A primitives ---------------------------------------------

    def _update(self, provided_data: bytes) -> None:
        h = self._hash
        k, v = self._key, self._v
        k = hmac.digest(k, v + b"\x00" + provided_data, h)
        v = hmac.digest(k, v, h)
        if provided_data:
            k = hmac.digest(k, v + b"\x01" + provided_data, h)
            v = hmac.digest(k, v, h)
        self._key, self._v = k, v

    def generate(self, n: int, additional_input: bytes = b"") -> bytes:
        """Return ``n`` bytes (at most 65536) as one SP 800-90A Generate call.

        Discards any buffered words so that the output is exactly what the
        standard specifies for the current state.
        """
        if not 0 <= n <= self._MAX_REQUEST_BYTES:
            raise ValueError(f"n must be between 0 and {self._MAX_REQUEST_BYTES}")
        if self._counter > self._RESEED_INTERVAL:
            raise RuntimeError("reseed required: reseed interval exceeded")
        additional_input = bytes(additional_input)
        if additional_input:
            self._update(additional_input)
        h, k, v = self._hash, self._key, self._v
        out = []
        have = 0
        while have < n:
            v = hmac.digest(k, v, h)
            out.append(v)
            have += len(v)
        self._v = v
        self._update(additional_input)
        self._counter += 1
        self._words = []
        return b"".join(out)[:n]

    # -- seeding -------------------------------------------------------------

    def seed(self, a: bytes | bytearray | memoryview | None = None) -> None:
        """Instantiate from ``entropy || nonce`` (>= 48 bytes), or OS entropy."""
        if a is None:
            material = os.urandom(self._MIN_SEED_BYTES)
        elif isinstance(a, (bytes, bytearray, memoryview)):
            material = bytes(a)
            if len(material) < self._MIN_SEED_BYTES:
                raise ValueError(
                    f"seed must be at least {self._MIN_SEED_BYTES} bytes "
                    f"(entropy + nonce), got {len(material)}"
                )
        else:
            raise TypeError(f"seed must be bytes or None, not {type(a).__name__}")
        size = self._DIGEST_SIZES[self._hash]
        self._key = b"\x00" * size
        self._v = b"\x01" * size
        self._update(material + self._personalization)
        self._counter = 1
        self._words = []

    def reseed(self, entropy: bytes, additional_input: bytes = b"") -> None:
        """Mix in at least 32 bytes of fresh entropy (SP 800-90A Reseed)."""
        entropy = bytes(entropy)
        if len(entropy) < self._STRENGTH_BYTES:
            raise ValueError(f"need at least {self._STRENGTH_BYTES} bytes of entropy")
        self._update(entropy + bytes(additional_input))
        self._counter = 1
        self._words = []

    # -- BitGenerator interface ----------------------------------------------

    def _refill(self) -> None:
        data = self.generate(self._REFILL_BYTES)
        self._words = list(reversed(struct.unpack(f"<{len(data) // 8}Q", data)))

    def random_raw(self) -> int:
        """Return the next 64-bit unsigned int."""
        words = self._words
        if not words:
            self._refill()
            words = self._words
        return words.pop()

    @property
    def state(self) -> dict[str, Any]:
        """Working state ``{key, v, reseed_counter, buffer}``; secret."""
        n = len(self._words)
        return {
            "key": self._key,
            "v": self._v,
            "reseed_counter": self._counter,
            "buffer": struct.pack(f"<{n}Q", *reversed(self._words)),
        }

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        key, v, counter, buffer = (
            value["key"], value["v"], value["reseed_counter"], value["buffer"],
        )
        size = self._DIGEST_SIZES[self._hash]
        if not (isinstance(key, bytes) and isinstance(v, bytes)
                and len(key) == len(v) == size):
            raise ValueError(f"state key and v must be {size} bytes for {self._hash}")
        if not isinstance(counter, int) or counter < 1:
            raise ValueError("state['reseed_counter'] must be a positive int")
        if (
            not isinstance(buffer, bytes)
            or len(buffer) % 8
            or len(buffer) > self._REFILL_BYTES
        ):
            raise ValueError(
                "state['buffer'] must be bytes, a multiple of 8 long, "
                f"at most {self._REFILL_BYTES}"
            )
        self._key, self._v, self._counter = key, v, counter
        self._words = list(reversed(struct.unpack(f"<{len(buffer) // 8}Q", buffer)))

    def __getstate__(self) -> dict[str, Any]:
        return {
            "hash": self._hash,
            "personalization": self._personalization,
            **self.state,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        if state["hash"] not in self._DIGEST_SIZES:
            raise ValueError(f"unknown hash {state['hash']!r}")
        self._hash = state["hash"]
        self._personalization = bytes(state["personalization"])
        self.state = state

    def __repr__(self) -> str:
        return f"{type(self).__name__}(hash={self._hash!r}, <secret state>)"


class SystemBitGenerator:
    """The operating system's CSPRNG, read in buffered chunks via ``os.urandom``.

    The analogue of :class:`random.SystemRandom`.  It has no seed and no
    state: ``Generator.seed``, ``getstate`` and ``setstate`` raise
    ``NotImplementedError``, and copies and unpickled instances draw fresh
    bytes rather than replaying the original's buffer.  The buffer is dropped
    in forked children so parent and child never return the same words.
    """

    __slots__ = ("_words", "__weakref__")

    _REFILL_BYTES = 1024
    _instances: weakref.WeakSet[SystemBitGenerator] = weakref.WeakSet()

    def __init__(self) -> None:
        self._words: list[int] = []
        self._instances.add(self)

    def random_raw(self) -> int:
        """Return the next 64-bit unsigned int."""
        words = self._words
        if not words:
            data = os.urandom(self._REFILL_BYTES)
            words = self._words = list(struct.unpack(f"<{len(data) // 8}Q", data))
        return words.pop()

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), ()

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    @classmethod
    def _after_fork_in_child(cls) -> None:
        for inst in list(cls._instances):
            inst._words = []


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=SystemBitGenerator._after_fork_in_child)
