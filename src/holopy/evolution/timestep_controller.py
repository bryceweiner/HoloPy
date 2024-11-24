from typing import Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from ..config.constants import (
    MIN_TIMESTEP,
    MAX_TIMESTEP,
    SAFETY_FACTOR,
    ERROR_TOLERANCE
)
import logging

logger = logging.getLogger(__name__)

@dataclass
class TimestepResult:
    """Contains timestep adaptation results."""
    dt: float
    error_estimate: float
    stability_measure: float
    accepted: bool

class AdaptiveTimestepController:
    """Controls evolution timestep adaptively."""
    
    def __init__(
        self,
        initial_dt: float,
        enable_predictor_corrector: bool = True
    ):
        self.dt = initial_dt
        self.enable_predictor_corrector = enable_predictor_corrector
        
        # Initialize controller state
        self.previous_error = None
        self.error_history = []
        self.dt_history = []
        
        logger.info(f"Initialized AdaptiveTimestepController with dt={initial_dt}")
    
    def calculate_timestep(
        self,
        current_state: np.ndarray,
        evolution_operator: callable,
        time: float
    ) -> TimestepResult:
        """Calculate optimal timestep for next evolution step."""
        try:
            # Estimate local error
            error_estimate = self._estimate_local_error(
                current_state,
                evolution_operator,
                time
            )
            
            # Calculate stability measure
            stability = self._calculate_stability_measure(
                current_state,
                evolution_operator,
                time
            )
            
            # Determine if step is acceptable
            accepted = error_estimate < ERROR_TOLERANCE
            
            if accepted:
                # Calculate new timestep
                new_dt = self._adapt_timestep(
                    error_estimate,
                    stability
                )
            else:
                # Reduce timestep for retry
                new_dt = self.dt * 0.5
            
            # Update controller state
            self.previous_error = error_estimate
            self.dt = new_dt
            
            # Update history
            self.error_history.append(error_estimate)
            self.dt_history.append(new_dt)
            
            return TimestepResult(
                dt=new_dt,
                error_estimate=error_estimate,
                stability_measure=stability,
                accepted=accepted
            )
            
        except Exception as e:
            logger.error(f"Timestep calculation failed: {str(e)}")
            raise
    
    def _estimate_local_error(
        self,
        state: np.ndarray,
        evolution_operator: callable,
        time: float
    ) -> float:
        """Estimate local truncation error."""
        try:
            if self.enable_predictor_corrector:
                # Predictor step (Euler)
                k1 = evolution_operator(state, time)
                predicted = state + self.dt * k1
                
                # Corrector step (Trapezoidal)
                k2 = evolution_operator(predicted, time + self.dt)
                corrected = state + 0.5 * self.dt * (k1 + k2)
                
                # Error estimate
                error = np.linalg.norm(corrected - predicted)
            else:
                # Simple error estimate using evolution operator
                k1 = evolution_operator(state, time)
                k2 = evolution_operator(state + self.dt * k1, time + self.dt)
                error = np.linalg.norm(k2 - k1) * self.dt
            
            return float(error)
            
        except Exception as e:
            logger.error(f"Error estimation failed: {str(e)}")
            raise
    
    def _calculate_stability_measure(
        self,
        state: np.ndarray,
        evolution_operator: callable,
        time: float
    ) -> float:
        """Calculate numerical stability measure."""
        try:
            # Calculate evolution operator eigenvalues
            k = evolution_operator(state, time)
            jacobian = k.reshape(-1, 1) @ state.reshape(1, -1)
            
            # Estimate spectral radius
            eigvals = np.linalg.eigvals(jacobian)
            spectral_radius = np.max(np.abs(eigvals))
            
            # Calculate stability measure
            stability = 1.0 / (1.0 + self.dt * spectral_radius)
            
            return float(stability)
            
        except Exception as e:
            logger.error(f"Stability calculation failed: {str(e)}")
            raise
    
    def _adapt_timestep(
        self,
        error: float,
        stability: float
    ) -> float:
        """Adapt timestep based on error and stability."""
        try:
            # Calculate optimal timestep scaling
            if error > 0:
                error_scaling = (ERROR_TOLERANCE / error)**0.5
            else:
                error_scaling = 2.0
            
            # Include stability consideration
            stability_scaling = stability
            
            # Combined scaling with safety factor
            scaling = min(
                error_scaling,
                stability_scaling
            ) * SAFETY_FACTOR
            
            # Calculate new timestep
            new_dt = self.dt * scaling
            
            # Apply bounds
            new_dt = np.clip(new_dt, MIN_TIMESTEP, MAX_TIMESTEP)
            
            return float(new_dt)
            
        except Exception as e:
            logger.error(f"Timestep adaptation failed: {str(e)}")
            raise 