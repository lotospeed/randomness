import copy
import json
import os
import pathlib
import pickle
import struct
import unittest

try:
    from Crypto.Cipher import AES as _aes
    from Crypto.Cipher import ChaCha20 as _cc
except ImportError:
    _aes = _cc = None

from randomness import Generator, SplitMix64
from randomness.bit_generators import HMAC_DRBG, SystemBitGenerator

if _cc is not None:
    from randomness.bit_generators import AESCounter, ChaCha20


def _rotl(x, n):
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


def reference_block(key, counter, nonce):
    """Independent pure-Python ChaCha20 block (original 64-bit counter layout)."""
    s = list(struct.unpack("<4I", b"expand 32-byte k"))
    s += struct.unpack("<8I", key)
    s += [counter & 0xFFFFFFFF, counter >> 32]
    s += struct.unpack("<2I", nonce)
    x = s[:]

    def qr(a, b, c, d):
        x[a] = (x[a] + x[b]) & 0xFFFFFFFF; x[d] = _rotl(x[d] ^ x[a], 16)
        x[c] = (x[c] + x[d]) & 0xFFFFFFFF; x[b] = _rotl(x[b] ^ x[c], 12)
        x[a] = (x[a] + x[b]) & 0xFFFFFFFF; x[d] = _rotl(x[d] ^ x[a], 8)
        x[c] = (x[c] + x[d]) & 0xFFFFFFFF; x[b] = _rotl(x[b] ^ x[c], 7)

    for _ in range(10):
        qr(0, 4, 8, 12); qr(1, 5, 9, 13); qr(2, 6, 10, 14); qr(3, 7, 11, 15)
        qr(0, 5, 10, 15); qr(1, 6, 11, 12); qr(2, 7, 8, 13); qr(3, 4, 9, 14)
    return struct.pack("<16I", *((a + b) & 0xFFFFFFFF for a, b in zip(x, s)))


# First 64 bytes of ChaCha20 keystream, all-zero key and nonce (DJB / RFC 7539 appendix).
ZERO_VECTOR = bytes.fromhex(
    "76b8e0ada0f13d90405d6ae55386bd28bdd219b8a08ded1aa836efcc8b770dc7"
    "da41597c5157488d7724e03fb8d84a376a43b8f41518a11cc387b669b2ee6586"
)


class _FastKeyErasureTests:
    """Behaviour shared by ChaCha20 and AESCounter; ``cls`` is the class under test."""

    def test_seeding(self):
        a, b = self.cls(bytes(32)), self.cls(bytes(32))
        self.assertEqual([a.random_raw() for _ in range(300)],
                         [b.random_raw() for _ in range(300)])
        self.assertNotEqual(self.cls().random_raw(), self.cls().random_raw())
        for bad in (b"short", bytes(33)):
            with self.assertRaises(ValueError):
                self.cls(bad)
        for bad in (123, "x" * 32):
            with self.assertRaises(TypeError):
                self.cls(bad)

    def test_reseed(self):
        a, b = self.cls(bytes(32)), self.cls(bytes(32))
        a.reseed(b"e" * 16)
        self.assertNotEqual(a.random_raw(), b.random_raw())
        a, b = self.cls(bytes(32)), self.cls(bytes(32))
        a.reseed(b"e" * 16)
        b.reseed(b"e" * 16)
        self.assertEqual(a.random_raw(), b.random_raw())
        c = self.cls(bytes(32))
        c.reseed(b"f" * 16)
        self.assertNotEqual(a.random_raw(), c.random_raw())
        with self.assertRaises(ValueError):
            a.reseed(b"short")

    def test_state_roundtrip_mid_buffer(self):
        g = self.cls(bytes(32))
        for _ in range(50):
            g.random_raw()
        st = g.state
        expected = [g.random_raw() for _ in range(300)]
        h = self.cls(bytes([1]) * 32)
        h.state = st
        self.assertEqual([h.random_raw() for _ in range(300)], expected)

    def test_state_validation(self):
        g = self.cls(bytes(32))
        for bad in ({"key": b"x", "buffer": b""},
                    {"key": bytes(32), "buffer": b"123"},
                    {"key": bytes(32), "buffer": bytes(8 * 125)}):
            with self.assertRaises(ValueError):
                g.state = bad

    def test_copy_and_pickle_are_independent(self):
        g = self.cls(bytes(32))
        g.random_raw()
        expected = _take(copy.deepcopy(g), 200)
        for c in (copy.copy(g), copy.deepcopy(g), pickle.loads(pickle.dumps(g))):
            self.assertIsNot(c, g)
            self.assertEqual(_take(c, 200), expected)
        self.assertEqual(_take(g, 200), expected)  # originals unaffected

    def test_repr_hides_state(self):
        g = self.cls(bytes(range(32)))
        self.assertNotIn(g.state["key"].hex(), repr(g))
        self.assertNotIn(str(g.state["key"]), repr(g))

    def test_with_generator(self):
        g = Generator(self.cls(bytes(32)))
        h = Generator(self.cls(bytes(32)))
        self.assertEqual([g.randrange(10**6) for _ in range(500)],
                         [h.randrange(10**6) for _ in range(500)])
        st = g.getstate()
        a = [g.gauss() for _ in range(5)]
        g.setstate(st)
        self.assertEqual(a, [g.gauss() for _ in range(5)])
        g.seed(bytes(32))
        self.assertEqual(g.random(), Generator(self.cls(bytes(32))).random())
        c = copy.copy(g)
        self.assertEqual(c.random(), g.random())


@unittest.skipIf(_cc is None, "pycryptodome not installed")
class TestChaCha20(_FastKeyErasureTests, unittest.TestCase):
    cls = property(lambda self: ChaCha20)

    def test_reference_block_matches_known_vector(self):
        self.assertEqual(reference_block(bytes(32), 0, bytes(8)), ZERO_VECTOR)

    def test_output_matches_independent_chacha(self):
        key = bytes(range(32))
        gen = ChaCha20(key)
        stream = b"".join(reference_block(key, i, bytes(8)) for i in range(16))
        expected = struct.unpack("<124Q", stream[32:])
        self.assertEqual([gen.random_raw() for _ in range(124)], list(expected))

    def test_key_erasure_chain(self):
        key = bytes(range(32))
        gen = ChaCha20(key)
        for _ in range(124):
            gen.random_raw()
        # Second refill runs under the first 32 keystream bytes of the first.
        key2 = b"".join(reference_block(key, i, bytes(8)) for i in range(1))[:32]
        stream = b"".join(reference_block(key2, i, bytes(8)) for i in range(16))
        self.assertEqual(gen.random_raw(), struct.unpack("<Q", stream[32:40])[0])


# FIPS 197 appendix C.3: AES-256, key 00..1f.
FIPS197_PLAIN = bytes.fromhex("00112233445566778899aabbccddeeff")
FIPS197_CIPHER = bytes.fromhex("8ea2b7ca516745bfeafc49904b496089")


def aes_ctr_reference(key, blocks):
    """AES-256-CTR keystream built block by block from single-block ECB calls."""
    ecb = _aes.new(key, _aes.MODE_ECB)
    return b"".join(ecb.encrypt(i.to_bytes(16, "big")) for i in range(blocks))


@unittest.skipIf(_cc is None, "pycryptodome not installed")
class TestAESCounter(_FastKeyErasureTests, unittest.TestCase):
    cls = property(lambda self: AESCounter)

    def test_reference_matches_known_vector(self):
        ecb = _aes.new(bytes(range(32)), _aes.MODE_ECB)
        self.assertEqual(ecb.encrypt(FIPS197_PLAIN), FIPS197_CIPHER)

    def test_known_answer(self):
        # AES-256 of counter block 2 under the all-zero key (the first
        # keystream block of NIST GCM test case 14); blocks 0-1 are the next key.
        gen = AESCounter(bytes(32))
        expected = struct.unpack("<2Q", bytes.fromhex("cea7403d4d606b6e074ec5d3baf39d18"))
        self.assertEqual([gen.random_raw(), gen.random_raw()], list(expected))

    def test_output_matches_independent_aes(self):
        key = bytes(range(32))
        gen = AESCounter(key)
        expected = struct.unpack("<124Q", aes_ctr_reference(key, 64)[32:])
        self.assertEqual([gen.random_raw() for _ in range(124)], list(expected))

    def test_key_erasure_chain(self):
        key = bytes(range(32))
        gen = AESCounter(key)
        for _ in range(124):
            gen.random_raw()
        # Second refill runs under the first 32 keystream bytes of the first.
        key2 = aes_ctr_reference(key, 2)
        stream = aes_ctr_reference(key2, 64)
        self.assertEqual(gen.random_raw(), struct.unpack("<Q", stream[32:40])[0])

    def test_differs_from_chacha20(self):
        self.assertNotEqual(_take(AESCounter(bytes(32)), 4), _take(ChaCha20(bytes(32)), 4))


def _take(bg, n):
    return [bg.random_raw() for _ in range(n)]


CAVP = json.loads((pathlib.Path(__file__).parent / "data" / "hmac_drbg_cavp.json").read_text())


class TestHMAC_DRBG(unittest.TestCase):
    def test_nist_cavp_vectors(self):
        """NIST CAVP HMAC_DRBG vectors (no prediction resistance, with reseed)."""
        self.assertEqual(len(CAVP), 24)
        for i, c in enumerate(CAVP):
            with self.subTest(i=i, hash=c["hash"]):
                n = len(c["returned"]) // 2
                d = HMAC_DRBG(
                    bytes.fromhex(c["entropy"]) + bytes.fromhex(c["nonce"]),
                    hash=c["hash"],
                    personalization=bytes.fromhex(c["personalization"]),
                )
                d.reseed(
                    bytes.fromhex(c["reseed_entropy"]),
                    bytes.fromhex(c["reseed_additional"]),
                )
                d.generate(n, bytes.fromhex(c["additional"][0]))  # discarded
                out = d.generate(n, bytes.fromhex(c["additional"][1]))
                self.assertEqual(out.hex(), c["returned"])

    def test_random_raw_is_buffered_generate(self):
        seed = bytes(range(48))
        a, b = HMAC_DRBG(seed), HMAC_DRBG(seed)
        stream = b.generate(1024) + b.generate(1024)
        expected = struct.unpack("<256Q", stream)
        self.assertEqual(_take(a, 256), list(expected))

    def test_seeding(self):
        self.assertEqual(_take(HMAC_DRBG(bytes(48)), 300), _take(HMAC_DRBG(bytes(48)), 300))
        self.assertNotEqual(HMAC_DRBG().random_raw(), HMAC_DRBG().random_raw())
        self.assertNotEqual(
            HMAC_DRBG(bytes(48)).random_raw(),
            HMAC_DRBG(bytes(48), personalization=b"x").random_raw(),
        )
        self.assertNotEqual(
            HMAC_DRBG(bytes(48)).random_raw(),
            HMAC_DRBG(bytes(48), hash="sha512").random_raw(),
        )
        with self.assertRaises(ValueError):
            HMAC_DRBG(bytes(47))
        with self.assertRaises(TypeError):
            HMAC_DRBG(12345)
        with self.assertRaises(ValueError):
            HMAC_DRBG(bytes(48), hash="md5")

    def test_reseed_and_limits(self):
        a, b = HMAC_DRBG(bytes(48)), HMAC_DRBG(bytes(48))
        a.reseed(b"e" * 32)
        self.assertNotEqual(a.random_raw(), b.random_raw())
        with self.assertRaises(ValueError):
            a.reseed(b"short")
        with self.assertRaises(ValueError):
            a.generate(65537)
        a._counter = HMAC_DRBG._RESEED_INTERVAL + 1
        with self.assertRaises(RuntimeError):
            a.generate(8)
        a.reseed(b"e" * 32)
        a.generate(8)
        fresh = HMAC_DRBG(bytes(48))
        fresh._counter = HMAC_DRBG._RESEED_INTERVAL + 1
        with self.assertRaises(RuntimeError):
            fresh.random_raw()

    def test_state_roundtrip_copy_pickle(self):
        for h in ("sha256", "sha512"):
            g = HMAC_DRBG(bytes(48), hash=h, personalization=b"p")
            _take(g, 50)
            expected = _take(copy.deepcopy(g), 300)
            for c in (copy.copy(g), copy.deepcopy(g), pickle.loads(pickle.dumps(g))):
                self.assertEqual(_take(c, 300), expected)
            fresh = HMAC_DRBG(bytes([1]) * 48, hash=h)
            fresh.state = g.state
            self.assertEqual(_take(fresh, 300), expected)

    def test_state_validation(self):
        g = HMAC_DRBG(bytes(48))
        good = g.state
        for bad in (
            {**good, "key": b"x"},
            {**good, "reseed_counter": 0},
            {**good, "buffer": b"123"},
            {**good, "buffer": bytes(2048)},
        ):
            with self.assertRaises(ValueError):
                g.state = bad

    def test_with_generator(self):
        g, h = Generator(HMAC_DRBG(bytes(48))), Generator(HMAC_DRBG(bytes(48)))
        self.assertEqual([g.randrange(10**6) for _ in range(500)],
                         [h.randrange(10**6) for _ in range(500)])
        st = g.getstate()
        a = [g.gauss() for _ in range(5)]
        g.setstate(st)
        self.assertEqual(a, [g.gauss() for _ in range(5)])
        g.seed(bytes(48))
        self.assertEqual(g.random(), Generator(HMAC_DRBG(bytes(48))).random())


class TestSystemBitGenerator(unittest.TestCase):
    def test_outputs(self):
        g = SystemBitGenerator()
        words = _take(g, 1000)
        self.assertTrue(all(0 <= w < 1 << 64 for w in words))
        self.assertEqual(len(set(words)), 1000)
        self.assertNotEqual(_take(SystemBitGenerator(), 5), _take(SystemBitGenerator(), 5))

    def test_no_seed_or_state(self):
        g = Generator(SystemBitGenerator())
        for f in (lambda: g.seed(1), g.getstate, lambda: g.setstate((3, {}, None))):
            with self.assertRaises(NotImplementedError):
                f()

    def test_copy_does_not_replay_buffer(self):
        g = SystemBitGenerator()
        g.random_raw()  # fill buffer
        upcoming = list(g._words)
        for c in (copy.copy(g), copy.deepcopy(g), pickle.loads(pickle.dumps(g))):
            self.assertEqual(c._words, [])
            self.assertTrue(set(_take(c, 20)).isdisjoint(upcoming))

    def test_with_generator(self):
        g = Generator(SystemBitGenerator())
        self.assertTrue(0.0 <= g.random() < 1.0)
        self.assertEqual(len(g.randbytes(100)), 100)
        self.assertIsNot(copy.copy(g).bit_generator, g.bit_generator)

    @unittest.skipUnless(hasattr(os, "fork"), "needs os.fork")
    def test_fork_does_not_duplicate_buffer(self):
        g = SystemBitGenerator()
        g.random_raw()
        parent_next = list(g._words)[::-1][:5]
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:  # child
            try:
                os.close(r)
                os.write(w, struct.pack("<5Q", *_take(g, 5)))
            finally:
                os._exit(0)
        os.close(w)
        child = struct.unpack("<5Q", os.read(r, 40))
        os.close(r)
        os.waitpid(pid, 0)
        self.assertEqual(_take(g, 5), parent_next)  # parent unaffected
        self.assertTrue(set(child).isdisjoint(parent_next))


if __name__ == "__main__":
    unittest.main()
