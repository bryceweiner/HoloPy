"""
Comprehensive validation suite for holographic simulation.
"""
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
from dataclasses import dataclass
from ..config.constants import (
    INFORMATION_GENERATION_RATE,
    PLANCK_CONSTANT,
    CRITICAL_THRESHOLD
)

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Container for validation results."""
    check_name: str
    passed: bool
    expected: float
    actual: float
    tolerance: float
    details: str

class HolographicValidationSuite:
    """Comprehensive validation suite for holographic simulations."""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.results_history: List[ValidationResult] = []
        
    def validate_simulation(
        self,
        current_state: np.ndarray,
        previous_state: Optional[np.ndarray],
        time: float,
        metadata: Dict
    ) -> List[ValidationResult]:
        """Run complete validation suite."""
        results = []
        
        # Information conservation check
        results.append(self._validate_information_conservation(
            current_state, previous_state, time
        ))
        
        # Cycle continuity check (from math.tex:2214-2216)
        if time >= 2/INFORMATION_GENERATION_RATE:
            results.append(self._validate_cycle_continuity(
                current_state, metadata.get('initial_state'), time
            ))
        
        # Unified framework validation (from math.tex:4971-4973)
        results.append(self._validate_unified_framework(
            current_state, time
        ))
        
        # Store and log results
        self.results_history.extend(results)
        self._log_validation_results(results)
        
        return results
    
    def _validate_information_conservation(
        self,
        current_state: np.ndarray,
        previous_state: Optional[np.ndarray],
        time: float
    ) -> ValidationResult:
        """Validate information conservation law."""
        if previous_state is None:
            return ValidationResult(
                check_name="information_conservation",
                passed=True,
                expected=0.0,
                actual=0.0,
                tolerance=self.tolerance,
                details="Initial state - no previous state for comparison"
            )
            
        current_info = self._calculate_information_content(current_state)
        previous_info = self._calculate_information_content(previous_state)
        
        expected_ratio = np.exp(-INFORMATION_GENERATION_RATE * time)
        actual_ratio = current_info / previous_info
        
        passed = np.abs(actual_ratio - expected_ratio) < self.tolerance
        
        return ValidationResult(
            check_name="information_conservation",
            passed=passed,
            expected=expected_ratio,
            actual=actual_ratio,
            tolerance=self.tolerance,
            details="Information conservation check"
        ) 