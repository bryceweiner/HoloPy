"""
HoloPy: Quantum Holographic Simulation Framework
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("holopy-quantum")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

from .core.hilbert import HilbertSpace
from .core.hilbert_continuum import HilbertContinuum, DualState
from .metrics.validation_suite import ValidationResult, HolographicValidationSuite
from .metrics.collectors import MetricsCollector

__all__ = [
    'HilbertSpace',
    'HilbertContinuum',
    'DualState',
    'ValidationResult',
    'HolographicValidationSuite',
    'MetricsCollector'
] 