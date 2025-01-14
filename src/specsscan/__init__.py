"""SpecsScan class easy access APIs"""
# Easy access APIs
import importlib.metadata

from .core import SpecsScan

__version__ = importlib.metadata.version("specsanalyzer")
__all__ = ["SpecsScan"]
