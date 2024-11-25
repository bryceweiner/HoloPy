"""
Holographic quantum noise implementation.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from ..config.constants import (
    INFORMATION_GENERATION_RATE,
    COUPLING_CONSTANT,
    PLANCK_CONSTANT,
    SPEED_OF_LIGHT,
    BOLTZMANN_CONSTANT,
    CRITICAL_THRESHOLD
)
import logging

logger = logging.getLogger(__name__)

@dataclass
class NoiseParameters:
    """Parameters for quantum noise models."""
    amplitude_damping: float
    phase_damping: float
    depolarizing: float
    thermal_noise: float
    correlation_length: float

class HolographicNoise:
    """Implements holographic quantum noise models."""
    
    def __init__(
        self,
        n_qubits: int,
        temperature: float = 0.0,
        correlation_time: float = 1e-12
    ):
        """
        Initialize noise model.
        
        Args:
            n_qubits: Number of qubits
            temperature: System temperature
            correlation_time: Noise correlation time
        """
        self.n_qubits = n_qubits
        self.system_size = 2**n_qubits
        self.temperature = temperature
        self.correlation_time = correlation_time
        
        # Initialize noise model
        self._initialize_noise()
        
        logger.info(
            f"Initialized HolographicNoise for {n_qubits} qubits at "
            f"T={temperature:.2e}K"
        )
    
    def _initialize_noise(self) -> None:
        """Initialize noise model parameters."""
        try:
            # Calculate base noise rates
            self.params = NoiseParameters(
                amplitude_damping=INFORMATION_GENERATION_RATE,
                phase_damping=INFORMATION_GENERATION_RATE * 2,
                depolarizing=INFORMATION_GENERATION_RATE / 4,
                thermal_noise=self.temperature / PLANCK_CONSTANT,
                correlation_length=SPEED_OF_LIGHT * self.correlation_time
            )
            
            # Initialize noise operators
            self._initialize_noise_operators()
            
        except Exception as e:
            logger.error(f"Noise initialization failed: {str(e)}")
            raise
    
    def _initialize_noise_operators(self) -> None:
        """Initialize quantum noise operators."""
        try:
            # Single-qubit Pauli operators
            I = np.eye(2)
            X = np.array([[0, 1], [1, 0]])
            Y = np.array([[0, -1j], [1j, 0]])
            Z = np.array([[1, 0], [0, -1]])
            
            # Initialize operator dictionary
            self.operators = {
                'I': I, 'X': X, 'Y': Y, 'Z': Z,
                'raise': np.array([[0, 1], [0, 0]]),
                'lower': np.array([[0, 0], [1, 0]])
            }
            
            # Initialize Kraus operators for each noise channel
            self._initialize_amplitude_damping()
            self._initialize_phase_damping()
            self._initialize_depolarizing()
            self._initialize_thermal_noise()
            
            logger.debug("Initialized noise operators")
            
        except Exception as e:
            logger.error(f"Operator initialization failed: {str(e)}")
            raise
    
    def _initialize_amplitude_damping(self) -> None:
        """Initialize amplitude damping channel."""
        gamma = self.params.amplitude_damping
        nth = self._thermal_occupation()
        
        # Ensure the argument to sqrt is non-negative
        damping_factor = np.clip(1 - (nth + 1) * gamma, 0, 1)
        
        self.amplitude_kraus = [
            np.array([[1, 0], [0, np.sqrt(damping_factor)]]),
            np.array([[0, np.sqrt(gamma)], [0, 0]])
        ]
    
    def _initialize_phase_damping(self) -> None:
        """Initialize phase damping channel."""
        lambda_pd = self.params.phase_damping
        self.phase_kraus = [
            np.array([[1, 0], [0, np.sqrt(1-lambda_pd)]]),
            np.array([[0, 0], [0, np.sqrt(lambda_pd)]])
        ]
    
    def _initialize_depolarizing(self) -> None:
        """Initialize depolarizing channel."""
        p = self.params.depolarizing
        self.depolarizing_kraus = [
            np.sqrt(1-3*p/4) * np.eye(2),
            np.sqrt(p/4) * self.operators['X'],
            np.sqrt(p/4) * self.operators['Y'],
            np.sqrt(p/4) * self.operators['Z']
        ]
    
    def _initialize_thermal_noise(self) -> None:
        """Initialize thermal noise channel."""
        nth = self._get_thermal_occupation()
        gamma = self.params.thermal_noise
        self.thermal_kraus = [
            np.sqrt(1 - (nth+1)*gamma) * np.eye(2),
            np.sqrt((nth+1)*gamma) * self.operators['lower'],
            np.sqrt(nth*gamma) * self.operators['raise']
        ]
    
    def _get_thermal_occupation(self) -> float:
        """Calculate thermal occupation number."""
        if self.temperature == 0:
            return 0.0
        energy = PLANCK_CONSTANT * SPEED_OF_LIGHT / self.params.correlation_length
        return 1.0 / (np.exp(energy / (BOLTZMANN_CONSTANT * self.temperature)) - 1.0)
    
    def apply_noise(self, state: np.ndarray, gamma: float) -> np.ndarray:
        """Apply noise to quantum state."""
        try:
            # Ensure gamma is in valid range
            gamma = np.clip(gamma, 0, 1)
            
            # Calculate thermal occupation
            nth = 0.1  # Default thermal occupation number
            
            # Safe sqrt calculation with clipping
            sqrt_term = np.clip(1 - (nth+1)*gamma, 0, 1)
            damping_op = np.sqrt(sqrt_term) * np.eye(2)
            
            noisy_state = state.copy()
            return damping_op @ noisy_state
        except Exception as e:
            logger.error(f"Noise application failed: {str(e)}")
            return state  # Return original state on error
        
    def _apply_amplitude_damping(
        self,
        state: np.ndarray,
        scale: float
    ) -> np.ndarray:
        """Apply amplitude damping channel."""
        result = np.zeros_like(state)
        for op in self.amplitude_kraus:
            expanded = self._expand_operator(op)
            result += expanded @ state
        return result
    
    def _apply_phase_damping(
        self,
        state: np.ndarray,
        scale: float
    ) -> np.ndarray:
        """Apply phase damping channel."""
        result = np.zeros_like(state)
        for op in self.phase_kraus:
            expanded = self._expand_operator(op)
            result += expanded @ state
        return result
    
    def _apply_depolarizing(
        self,
        state: np.ndarray,
        scale: float
    ) -> np.ndarray:
        """Apply depolarizing channel."""
        result = np.zeros_like(state)
        for op in self.depolarizing_kraus:
            expanded = self._expand_operator(op)
            result += expanded @ state
        return result
    
    def _apply_thermal_noise(
        self,
        state: np.ndarray,
        scale: float
    ) -> np.ndarray:
        """Apply thermal noise channel."""
        result = np.zeros_like(state)
        for op in self.thermal_kraus:
            expanded = self._expand_operator(op)
            result += expanded @ state
        return result
    
    def _apply_holographic_corrections(
        self,
        state: np.ndarray
    ) -> np.ndarray:
        """Apply holographic corrections to noise channels."""
        # Transform to k-space
        k_space = np.fft.fft(state)
        
        # Calculate correction factors
        k_values = 2 * np.pi * np.fft.fftfreq(self.system_size)
        corrections = np.exp(-COUPLING_CONSTANT * np.abs(k_values))
        
        # Apply corrections
        k_space *= corrections
        
        # Transform back
        return np.fft.ifft(k_space)
    
    def _expand_operator(self, operator: np.ndarray) -> np.ndarray:
        """Expand single-qubit operator to full system size."""
        result = np.eye(1)
        for _ in range(self.n_qubits):
            result = np.kron(result, operator)
        return result
    
    def _generate_lindblad_operators(self) -> List[np.ndarray]:
        """Generate Lindblad operators for the noise model."""
        gamma = self.dephasing_rate
        nth = self._thermal_occupation()
        
        # Ensure valid sqrt arguments
        decay_factor = np.clip(1 - (nth + 1) * gamma, 0, 1)  # Clip to [0,1] range
        
        operators = [
            np.sqrt(decay_factor) * np.eye(2),
            np.sqrt(np.clip(gamma * (nth + 1), 0, 1)) * self._sigma_minus(),
            np.sqrt(np.clip(gamma * nth, 0, 1)) * self._sigma_plus()
        ]
        
        return operators