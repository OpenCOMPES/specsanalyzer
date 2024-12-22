"""SpecsAnalyzer class easy access APIs"""
# Easy access APIs
import importlib.metadata

from .core import SpecsAnalyzer

__version__ = importlib.metadata.version("specsanalyzer")
__all__ = ["SpecsAnalyzer"]
