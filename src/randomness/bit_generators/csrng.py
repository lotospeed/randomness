"""Cryptographically secure bit generators.

* :class:`SystemBitGenerator` -- the operating system's CSPRNG (``os.urandom``).
* :class:`HMAC_DRBG` -- NIST SP 800-90A HMAC_DRBG (standard library only).
* :class:`ChaCha20`, :class:`AESCounter` -- ChaCha20 / AES-256-CTR with fast
  key erasure; need the optional ``pycryptodome`` dependency
  (``pip install 'randomness[crypto]'``).

The seeded generators are deterministic: they are only as secret as their
seed, and their ``state`` / ``getstate`` / ``copy`` / ``pickle`` output is
key material.  Like any userspace PRNG they also duplicate their state across
``os.fork``; reseed in the child if that matters.
"""

from ._csrng import AESCounter, ChaCha20, HMAC_DRBG, SystemBitGenerator

__all__ = ["AESCounter", "ChaCha20", "HMAC_DRBG", "SystemBitGenerator"]
