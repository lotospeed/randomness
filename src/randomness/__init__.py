"""randomness: random.Random-style generators driven by explicit bit generators."""

from ._bitgen import BitGenerator
from ._generator import Generator
from .bit_generators.prng import SplitMix64

__all__ = ["BitGenerator", "Generator", "SplitMix64", "main"]


def main() -> None:
    print("Hello from randomness!")
