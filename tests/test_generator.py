import _random
import copy
import math
import pickle
import random
import unittest
from collections import Counter

from randomness import BitGenerator, Generator, SplitMix64


class Counting:
    """Bit generator wrapper that counts raw words consumed."""

    def __init__(self, seed=1):
        self.inner = SplitMix64(seed)
        self.calls = 0

    def random_raw(self):
        self.calls += 1
        return self.inner.random_raw()


class Replay:
    """Bit generator that replays a fixed list of words (and is unhashable)."""

    def __init__(self, words):
        self.words = list(words)

    __eq__ = object.__eq__
    __hash__ = None

    def random_raw(self):
        return self.words.pop(0)


def exercise(g):
    """Call every public Random method; return the outputs."""
    pop = list(range(20))
    out = [
        g.random(), g.getrandbits(1), g.getrandbits(64), g.getrandbits(200),
        g.randbytes(13), g.randrange(10), g.randrange(3, 100, 7),
        g.randrange(10**30), g.randint(-5, 5), g.choice(pop),
        g.sample(pop, 5), g.sample(pop, 3, counts=[2] * 20),
        g.choices(pop, k=4), g.choices(pop, weights=range(20), k=4),
        g.uniform(1, 2), g.triangular(0, 5, 1), g.normalvariate(0, 1),
        g.gauss(0, 1), g.gauss(0, 1), g.lognormvariate(0, 1),
        g.expovariate(2.0), g.vonmisesvariate(1.0, 4.0),
        g.vonmisesvariate(0.0, 1e-9), g.gammavariate(0.5, 1.0),
        g.gammavariate(1.0, 2.0), g.gammavariate(3.0, 1.0),
        g.betavariate(2, 3), g.paretovariate(3.0), g.weibullvariate(1, 2),
        g.binomialvariate(10, 0.3), g.binomialvariate(1000, 0.5),
    ]
    s = pop[:]
    g.shuffle(s)
    out.append(s)
    return out


class TestPrimitives(unittest.TestCase):
    def test_random_uses_high_53_bits(self):
        w = 0xFFFFFFFFFFFFFFFF
        g = Generator(Replay([w, 0, 1 << 11]))
        self.assertEqual(g.random(), 1 - 2.0**-53)
        self.assertEqual(g.random(), 0.0)
        self.assertEqual(g.random(), 2.0**-53)

    def test_getrandbits_layout(self):
        a, b, c = 0x0123456789ABCDEF, 0xFEDCBA9876543210, 0xF000000000000000
        self.assertEqual(Generator(Replay([a])).getrandbits(64), a)
        self.assertEqual(Generator(Replay([a])).getrandbits(8), 0x01)
        self.assertEqual(Generator(Replay([a, b])).getrandbits(128), a | b << 64)
        self.assertEqual(Generator(Replay([a, b, c])).getrandbits(132), a | b << 64 | 0xF << 128)

    def test_getrandbits_edges(self):
        g = Generator(Counting())
        self.assertEqual(g.getrandbits(0), 0)
        self.assertEqual(g.bit_generator.calls, 0)
        with self.assertRaises(ValueError):
            g.getrandbits(-1)
        for k in (1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000):
            for _ in range(20):
                self.assertLess(g.getrandbits(k), 1 << k)

    def test_words_consumed(self):
        g = Generator(Counting())
        g.getrandbits(64); self.assertEqual(g.bit_generator.calls, 1)
        g.getrandbits(65); self.assertEqual(g.bit_generator.calls, 3)
        g.random(); self.assertEqual(g.bit_generator.calls, 4)

    def test_bit_balance(self):
        g = Generator(SplitMix64(7))
        n = 20000
        for k in (1, 7, 64, 100):
            ones = [0] * k
            for _ in range(n):
                v = g.getrandbits(k)
                for i in range(k):
                    ones[i] += v >> i & 1
            for c in ones:
                self.assertLess(abs(c - n / 2), 5 * math.sqrt(n / 4))


class TestFullSuite(unittest.TestCase):
    def test_no_hidden_mersenne_twister(self):
        """Scrambling the base class's C state must not change any output."""
        a, b = Generator(SplitMix64(42)), Generator(SplitMix64(42))
        _random.Random.seed(b, 999)
        mt_before = _random.Random.getstate(b)
        self.assertEqual(exercise(a), exercise(b))
        self.assertEqual(_random.Random.getstate(b), mt_before)

    def test_every_method_consumes_raw_words(self):
        g = Generator(Counting())
        pop = list(range(10))
        calls = [
            lambda: g.random(), lambda: g.getrandbits(5), lambda: g.randbytes(3),
            lambda: g.randrange(7), lambda: g.randint(1, 6), lambda: g.choice(pop),
            lambda: g.shuffle(pop), lambda: g.sample(pop, 2),
            lambda: g.choices(pop), lambda: g.uniform(0, 1),
            lambda: g.triangular(), lambda: g.normalvariate(), lambda: g.gauss(),
            lambda: g.lognormvariate(0, 1), lambda: g.expovariate(),
            lambda: g.vonmisesvariate(0, 1), lambda: g.gammavariate(2, 1),
            lambda: g.betavariate(1, 1), lambda: g.paretovariate(1),
            lambda: g.weibullvariate(1, 1), lambda: g.binomialvariate(5, 0.5),
        ]
        for f in calls:
            before = g.bit_generator.calls
            f()
            self.assertGreater(g.bit_generator.calls, before)

    def test_distribution_sanity(self):
        g = Generator(SplitMix64(3))
        n = 20000
        xs = [g.gauss(5, 2) for _ in range(n)]
        mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / n
        self.assertAlmostEqual(mean, 5, delta=0.1)
        self.assertAlmostEqual(var, 4, delta=0.3)
        counts = Counter(g.randrange(6) for _ in range(n))
        for face in range(6):
            self.assertAlmostEqual(counts[face] / n, 1 / 6, delta=0.02)

    def test_is_a_random(self):
        self.assertIsInstance(Generator(SplitMix64(0)), random.Random)


class TestState(unittest.TestCase):
    def test_getstate_setstate(self):
        g = Generator(SplitMix64(1))
        g.gauss()  # leaves gauss_next populated
        st = g.getstate()
        expected = exercise(g)
        g.setstate(st)
        self.assertEqual(exercise(g), expected)

    def test_setstate_bad_version(self):
        g = Generator(SplitMix64(1))
        _, s, n = g.getstate()
        with self.assertRaises(ValueError):
            g.setstate((99, s, n))

    def test_seed_delegates_and_resets_gauss(self):
        g = Generator(SplitMix64(1))
        g.gauss()
        self.assertIsNotNone(g.gauss_next)
        g.seed(5)
        self.assertIsNone(g.gauss_next)
        self.assertEqual(g.bit_generator.state, SplitMix64(5).state)
        self.assertEqual(g.random(), Generator(SplitMix64(5)).random())

    def test_seed_unsupported(self):
        with self.assertRaises(NotImplementedError):
            Generator(Replay([1])).seed(1)

    def test_state_unsupported(self):
        with self.assertRaises(NotImplementedError):
            Generator(Replay([1])).getstate()

    def test_copy_is_independent(self):
        g = Generator(SplitMix64(9))
        g.gauss()
        c = copy.copy(g)
        self.assertIsNot(c.bit_generator, g.bit_generator)
        self.assertEqual(exercise(c), exercise(g))

    def test_deepcopy_and_pickle(self):
        g = Generator(SplitMix64(9))
        g.gauss()
        for c in (copy.deepcopy(g), pickle.loads(pickle.dumps(g))):
            self.assertIsNot(c.bit_generator, g.bit_generator)
            self.assertEqual(c.gauss_next, g.gauss_next)
            self.assertEqual(exercise(c), exercise(copy.deepcopy(g)))


class TestConstruction(unittest.TestCase):
    def test_unhashable_bit_generator(self):
        Generator(Replay([1]))

    def test_rejects_non_bitgenerator(self):
        with self.assertRaises(TypeError):
            Generator(object())

    def test_protocol(self):
        self.assertIsInstance(SplitMix64(), BitGenerator)

    def test_does_not_reseed_bit_generator(self):
        bg = SplitMix64(123)
        Generator(bg)
        self.assertEqual(bg.state, 123)


class TestSplitMix64(unittest.TestCase):
    def test_reference_vector(self):
        # Reference outputs for seed 0 (from the canonical C implementation).
        bg = SplitMix64(0)
        self.assertEqual(
            [bg.random_raw() for _ in range(3)],
            [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F],
        )

    def test_outputs_are_64_bit(self):
        bg = SplitMix64(None)
        for _ in range(100):
            self.assertTrue(0 <= bg.random_raw() < 1 << 64)

    def test_seed_validation(self):
        with self.assertRaises(TypeError):
            SplitMix64("x")
        self.assertEqual(SplitMix64(-1).state, (1 << 64) - 1)


class TestNumpyInterop(unittest.TestCase):
    def test_numpy_bit_generators(self):
        np = __import__("numpy", fromlist=["random"])
        for cls in (np.random.PCG64, np.random.Philox, np.random.SFC64):
            g = Generator(cls(1))
            self.assertIsInstance(g.random(), float)
            ref = cls(1)
            self.assertEqual(Generator(ref).getrandbits(64), cls(1).random_raw())
            exercise(g)
            st = g.getstate()
            a = exercise(g)
            g.setstate(st)
            self.assertEqual(exercise(g), a)
            with self.assertRaises(NotImplementedError):
                g.seed(5)  # numpy bit generators are reseeded by rebuilding


try:
    import numpy  # noqa: F401
except ImportError:
    del TestNumpyInterop

if __name__ == "__main__":
    unittest.main()
