import copy
import pickle
import unittest

import numpy as np

from randomness import BitGenerator, Generator, SplitMix64
from randomness.bit_generators.prng import PCG64, SFC64, Philox
from randomness.bit_generators import prng

CASES = [(PCG64, np.random.PCG64), (Philox, np.random.Philox), (SFC64, np.random.SFC64)]


def take(bg, n):
    return [bg.random_raw() for _ in range(n)]


class TestNumpyWrappers(unittest.TestCase):
    def test_matches_numpy_stream(self):
        for ours, theirs in CASES:
            with self.subTest(ours.__name__):
                n = 3000  # spans several refills
                expected = theirs(12345).random_raw(n).tolist()
                self.assertEqual(take(ours(12345), n), expected)

    def test_seed_forms(self):
        for ours, theirs in CASES:
            with self.subTest(ours.__name__):
                ss = np.random.SeedSequence([1, 2, 3])
                self.assertEqual(take(ours(ss), 5), theirs(np.random.SeedSequence([1, 2, 3])).random_raw(5).tolist())
                self.assertNotEqual(take(ours(), 3), take(ours(), 3))
                with self.assertRaises(ValueError):
                    ours(-1)

    def test_reseed(self):
        for ours, _ in CASES:
            with self.subTest(ours.__name__):
                g = ours(1)
                take(g, 10)
                g.seed(2)
                self.assertEqual(take(g, 50), take(ours(2), 50))

    def test_state_roundtrip_mid_buffer(self):
        for ours, _ in CASES:
            with self.subTest(ours.__name__):
                g = ours(7)
                take(g, 500)
                st = g.state
                expected = take(g, 2500)
                h = ours(99)
                h.state = st
                self.assertEqual(take(h, 2500), expected)

    def test_state_validation(self):
        g = PCG64(1)
        good = g.state
        for bad in ({**good, "buffer": b"123"}, {**good, "buffer": bytes(8 * 1025)}):
            with self.assertRaises(ValueError):
                g.state = bad
        with self.assertRaises(ValueError):  # numpy rejects another algorithm's state
            g.state = {**good, "bit_generator": SFC64(1).state}

    def test_copy_and_pickle_are_independent(self):
        for ours, _ in CASES:
            with self.subTest(ours.__name__):
                g = ours(3)
                take(g, 17)
                expected = take(copy.deepcopy(g), 2200)
                for c in (copy.copy(g), copy.deepcopy(g), pickle.loads(pickle.dumps(g))):
                    self.assertIsNot(c, g)
                    self.assertEqual(take(c, 2200), expected)
                self.assertEqual(take(g, 2200), expected)

    def test_outputs_are_64_bit_and_protocol(self):
        for ours, _ in CASES:
            g = ours(5)
            self.assertIsInstance(g, BitGenerator)
            self.assertTrue(all(0 <= w < 1 << 64 for w in take(g, 2000)))
            self.assertGreater(max(take(g, 2000)).bit_length(), 62)

    def test_with_generator(self):
        for ours, _ in CASES:
            with self.subTest(ours.__name__):
                g, h = Generator(ours(8)), Generator(ours(8))
                self.assertEqual([g.randrange(10**6) for _ in range(500)],
                                 [h.randrange(10**6) for _ in range(500)])
                st = g.getstate()
                a = [g.gauss() for _ in range(5)] + g.sample(range(100), 5)
                g.setstate(st)
                self.assertEqual(a, [g.gauss() for _ in range(5)] + g.sample(range(100), 5))
                g.seed(8)
                self.assertEqual(g.random(), Generator(ours(8)).random())
                c = copy.copy(g)
                self.assertEqual(c.random(), g.random())

    def test_mt19937_is_not_offered(self):
        self.assertFalse(hasattr(prng, "MT19937"))


class TestLayout(unittest.TestCase):
    def test_sections(self):
        from randomness import bit_generators as bg
        from randomness.bit_generators import csrng

        self.assertIs(bg.prng, prng)
        self.assertIs(bg.csrng, csrng)
        # Only the two sections are public; generators live inside them.
        self.assertEqual([n for n in dir(bg) if not n.startswith("__")], ["csrng", "prng"])
        self.assertIs(SplitMix64, prng.SplitMix64)  # top-level compat

    def test_import_does_not_pull_in_optional_deps(self):
        import pathlib
        import subprocess
        import sys

        src = str(pathlib.Path(__file__).parents[1] / "src")
        code = (f"import sys; sys.path.insert(0, {src!r});"
                "import randomness, randomness.bit_generators as b;"
                "b.csrng.HMAC_DRBG(); b.prng.SplitMix64(1);"
                "print('numpy' in sys.modules, 'Crypto' in sys.modules)")
        out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True,
                             text=True, check=True).stdout
        self.assertEqual(out.strip(), "False False")


if __name__ == "__main__":
    unittest.main()
