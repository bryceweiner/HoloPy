"""
Quantum state management module for the holographic framework.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from ..config.constants import (
    INFORMATION_GENERATION_RATE,
    PLANCK_CONSTANT,
    SPEED_OF_LIGHT
)

@dataclass
class QuantumState:
    """Represents a quantum state in the holographic framework."""
    
    wavefunction: np.ndarray
    position: np.ndarray
    momentum: np.ndarray
    time: float
    phase: float
    coherence: float
    
    @classmethod
    def create_initial_state(
        cls,
        grid_points: int,
        spatial_extent: float
    ) -> 'QuantumState':
        """Create an initial quantum state with a Gaussian wavepacket."""
        x = np.linspace(-spatial_extent/2, spatial_extent/2, grid_points)
        p = 2 * np.pi * np.fft.fftfreq(grid_points, x[1]-x[0])
        
        # Create Gaussian wavepacket
        psi = np.exp(-x**2 / (2 * spatial_extent/10)**2)
        psi = psi / np.sqrt(np.sum(np.abs(psi)**2))
        
        return cls(
            wavefunction=psi,
            position=x,
            momentum=p,
            time=0.0,
            phase=0.0,
            coherence=1.0
        )
    
    def evolve(self, dt: float) -> None:
        """Evolve the quantum state forward in time."""
        # Apply modified Schrödinger equation with information generation rate
        self.wavefunction *= np.exp(-INFORMATION_GENERATION_RATE * dt / 2)
        self.time += dt
        self.update_coherence()
    
    def update_coherence(self) -> None:
        """Update the coherence measure of the state."""
        self.coherence = np.exp(-INFORMATION_GENERATION_RATE * self.time)
    
    def calculate_observables(self) -> Tuple[float, float, float]:
        """Calculate expectation values of key observables."""
        density = np.abs(self.wavefunction)**2
        x_expect = np.sum(self.position * density)
        p_expect = np.sum(self.momentum * np.abs(np.fft.fft(self.wavefunction))**2)
        energy = np.sum(0.5 * self.momentum**2 * np.abs(np.fft.fft(self.wavefunction))**2)
        
        return x_expect, p_expect, energy 