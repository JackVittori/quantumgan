"""Package containing QGAN module architecture."""

from .discriminator import Discriminator
from .generator import PatchQuantumGenerator

__all__ = ["Discriminator",
           "PatchQuantumGenerator"
           ]
__version__ = '0.1'
