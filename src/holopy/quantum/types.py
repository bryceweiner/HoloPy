"""Type definitions for quantum operations."""
from enum import Enum, auto
from dataclasses import dataclass
from typing import Set, Dict, Optional, Protocol
import numpy as np

class ChannelType(Enum):
    """Types of quantum channels."""
    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"
    PAULI = "pauli"
    THERMAL = "thermal"
    RESET = "reset"
    CUSTOM = "custom"

class ErrorMitigationType(Enum):
    """Types of error mitigation strategies."""
    NONE = "none"
    RICHARDSON = "richardson"
    ZERO_NOISE = "zero_noise"
    PROBABILISTIC = "probabilistic"
    SYMMETRY = "symmetry"
    CUSTOM = "custom"

class ErrorMitigation(Protocol):
    """Protocol for error mitigation strategies."""
    def mitigate_error(
        self,
        state: np.ndarray,
        error_params: Optional[Dict] = None
    ) -> np.ndarray:
        """Mitigate errors in quantum state."""
        ...

@dataclass
class DefaultErrorMitigation:
    """Default error mitigation implementation."""
    mitigation_type: ErrorMitigationType = ErrorMitigationType.NONE
    
    def mitigate_error(
        self,
        state: np.ndarray,
        error_params: Optional[Dict] = None
    ) -> np.ndarray:
        """Default error mitigation (no operation)."""
        return state.copy() 