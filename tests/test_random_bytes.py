import copy
import random
import struct
import unittest

from randomness import Generator, SplitMix64
from randomness.bit_generators.csrng import HMAC_DRBG, SystemBitGenerator

try:
    from randomness.bit_generators.csrng import AESCounter, ChaCha20
    AESCounter(bytes(32))
except ImportError:
    AESCounter = ChaCha20 = None

try:
    from randomness.bit_generators.prng import PCG64, SFC64, Philox
    PCG64(0)
except ImportError:
    PCG64 = SFC64 = Philox = None

# Sizes around word, buffer and refill boundaries.
SIZES = [0, 1, 7, 8, 9, 16, 100, 991, 992, 993, 1024, 8192, 8200, 20001, 3, 0, 5000]


def _seeded():
    """Fresh, identically seeded bit generators that have a fast byte path."""
    made = [HMAC_DRBG(bytes(48)), HMAC_DRBG(bytes(48), hash="sha512")]
    if ChaCha20 is not None:
        made += [ChaCha20(bytes(32)), AESCounter(bytes(32))]
    if PCG64 is not None:
        made += [PCG64(7), Philox(7), SFC64(7)]
    return made


class WordsOnly:
    """Hides ``random_bytes`` so that Generator takes the inherited path."""

    def __init__(self, bit_generator):
        self.random_raw = bit_generator.random_raw


class TestRandomBytes(unittest.TestCase):
    def test_is_the_word_stream(self):
        for fast, slow in zip(_seeded(), _seeded()):
            with self.subTest(type(fast).__name__):
                fast.random_raw(), slow.random_raw()  # start mid-buffer
                for n in SIZES:
                    words = [slow.random_raw() for _ in range(-(-n // 8))]
                    self.assertEqual(fast.random_bytes(n),
                                     struct.pack(f"<{len(words)}Q", *words)[:n])
                    self.assertEqual(fast.random_raw(), slow.random_raw())

    def test_negative(self):
        for bg in _seeded() + [SystemBitGenerator()]:
            with self.assertRaises(ValueError):
                bg.random_bytes(-1)

    def test_state_roundtrip_after_bytes(self):
        for bg in _seeded():
            bg.random_bytes(1500)
            twin = copy.deepcopy(bg)
            self.assertEqual(twin.random_bytes(3000), bg.random_bytes(3000))

    def test_system(self):
        bg = SystemBitGenerator()
        for n in (0, 1, 8, 5000):
            self.assertEqual(len(bg.random_bytes(n)), n)
        self.assertNotEqual(bg.random_bytes(32), bg.random_bytes(32))


class TestGeneratorRandbytes(unittest.TestCase):
    def test_same_output_as_inherited_path(self):
        for fast, slow in zip(_seeded(), _seeded()):
            with self.subTest(type(fast).__name__):
                g, h = Generator(fast), Generator(WordsOnly(slow))
                for n in SIZES:
                    self.assertEqual(g.randbytes(n), h.randbytes(n))
                    self.assertEqual(g.random(), h.random())

    def test_getrandbits_same_output_as_word_path(self):
        ks = [0, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 319, 320, 1000, 1024,
              4096, 7937, 8000, 70001, 200]
        for fast, slow in zip(_seeded(), _seeded()):
            with self.subTest(type(fast).__name__):
                g, h = Generator(fast), Generator(WordsOnly(slow))
                for k in ks:
                    self.assertEqual(g.getrandbits(k), h.getrandbits(k))
                    self.assertEqual(g.random(), h.random())
                n = (1 << 2048) - 12345
                self.assertEqual([g.randrange(n) for _ in range(20)],
                                 [h.randrange(n) for _ in range(20)])

    def test_getrandbits_range(self):
        g = Generator(HMAC_DRBG(bytes(48)))
        for k in (65, 128, 1000):
            top = 0
            for _ in range(200):
                v = g.getrandbits(k)
                self.assertTrue(0 <= v < 1 << k)
                top |= v
            self.assertEqual(top.bit_length(), k)

    def test_without_fast_path(self):
        g = Generator(SplitMix64(1))
        self.assertEqual(g.randbytes(13), random.Random.randbytes(Generator(SplitMix64(1)), 13))

    def test_arguments(self):
        for g in (Generator(HMAC_DRBG(bytes(48))), Generator(SplitMix64(1))):
            with self.assertRaises(ValueError):
                g.randbytes(-1)
            with self.assertRaises(TypeError):
                g.randbytes(1.5)
            self.assertEqual(g.randbytes(0), b"")
            self.assertEqual(len(g.randbytes(True)), 1)

    def test_system(self):
        g = Generator(SystemBitGenerator())
        self.assertEqual([len(g.randbytes(n)) for n in (0, 3, 8, 1001)], [0, 3, 8, 1001])


if __name__ == "__main__":
    unittest.main()
