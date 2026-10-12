# randomness

`random.Random` driven by an explicit bit generator, the way
`numpy.random.Generator` is driven by a `BitGenerator`.

```python
from randomness import Generator, SplitMix64

g = Generator(SplitMix64(42))
g.random(); g.randrange(10); g.shuffle(xs); g.gauss(0, 1); g.sample(xs, 3)
```

`Generator` subclasses `random.Random`, so the whole stdlib API is available,
but every value is derived solely from `bit_generator.random_raw()` (a uniform
64-bit unsigned int). Any object with that method works, including numpy's
`PCG64`, `Philox`, etc. Optional `seed()` / `state` members on the bit
generator enable `Generator.seed`, `getstate`/`setstate`, `copy` and `pickle`.

## Bit generators

`randomness.bit_generators` has two sections, both re-exported from the package:

* `randomness.bit_generators.prng`: fast and statistically strong, **not** secure
* `randomness.bit_generators.csrng`: cryptographically secure

### prng

```python
from randomness.bit_generators import SplitMix64, PCG64, Philox, SFC64

g = Generator(PCG64(42))        # int, SeedSequence, or None for OS entropy
```

`PCG64`, `Philox` and `SFC64` wrap numpy's implementations (`pip install
'randomness[numpy]'`): the stream is exactly numpy's `random_raw`, but they add
`seed()`, `state`, `copy`/`pickle` support, and batched reads (~1.5x faster
than calling numpy once per word). `SplitMix64` is pure Python. MT19937 is not
offered: numpy's `random_raw` returns 32-bit values for it, which breaks the
64-bit contract.

### csrng

```python
from randomness import Generator
from randomness.bit_generators import ChaCha20, HMAC_DRBG, SystemBitGenerator

g = Generator(SystemBitGenerator())          # the OS CSPRNG; no seed or state
g = Generator(HMAC_DRBG())                   # NIST SP 800-90A, stdlib only
g = Generator(HMAC_DRBG(seed48, hash="sha512", personalization=b"app"))
g = Generator(ChaCha20(b"k" * 32))           # needs the 'crypto' extra
```

| class | seed | needs | notes |
|---|---|---|---|
| `SystemBitGenerator` | none | stdlib | like `random.SystemRandom`; buffer dropped on `fork` |
| `HMAC_DRBG` | >= 48 bytes (entropy + nonce) or `None` | stdlib | passes NIST CAVP vectors; `reseed`, `generate` |
| `ChaCha20` | exactly 32 bytes or `None` | `pycryptodome` (`pip install 'randomness[crypto]'`) | fast key erasure; `reseed` |

The seeded generators are deterministic, so they are only as secret as their
seed. Their `state`, and therefore `getstate()`, `copy` and `pickle`, contain
key material: keep it secret. Like any userspace PRNG they duplicate their
state across `os.fork`; reseed in the child if that matters.
