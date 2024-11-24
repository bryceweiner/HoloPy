"""
Classical observable management for holographic simulation.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
from ..config.constants import (
    INFORMATION_GENERATION_RATE,
    BOLTZMANN_CONSTANT,
    PLANCK_CONSTANT
)

@dataclass
class ContinuumState:
    """Container for classical observable state."""
    time: float
    density: np.ndarray
    temperature: float
    entropy: float
    information_content: float
    
    @classmethod
    def from_quantum_state(
        cls,
        quantum_state: np.ndarray,
        time: float,
        spatial_grid: np.ndarray
    ) -> 'ContinuumState':
        """Create classical state from quantum state."""
        density = np.abs(quantum_state)**2
        
        # Calculate temperature using equation from knowledgebase.md:750-761
        coherence = np.abs(np.vdot(quantum_state, quantum_state))
        temperature = -PLANCK_CONSTANT * INFORMATION_GENERATION_RATE / (
            2 * np.pi * BOLTZMANN_CONSTANT * np.log(coherence)
        )
        
        # Calculate entropy with holographic corrections
        entropy = -np.sum(density * np.log2(density + 1e-10))
        entropy *= np.exp(-INFORMATION_GENERATION_RATE * time)
        
        # Calculate information content
        information = cls._calculate_information_content(density, time)
        
        return cls(
            time=time,
            density=density,
            temperature=temperature,
            entropy=entropy,
            information_content=information
        )
    
    @staticmethod
    def _calculate_information_content(
        density: np.ndarray,
        time: float
    ) -> float:
        """Calculate classical information content."""
        raw_info = -np.sum(density * np.log2(density + 1e-10))
        return raw_info * np.exp(-INFORMATION_GENERATION_RATE * time)
    
    def evolve(self, dt: float) -> None:
        """Evolve classical state forward in time."""
        # Update time
        self.time += dt
        
        # Apply holographic decay to observables
        decay_factor = np.exp(-INFORMATION_GENERATION_RATE * dt)
        self.density *= decay_factor
        self.entropy *= decay_factor
        self.information_content *= decay_factor
        
        # Update temperature using modified Stefan-Boltzmann law
        self.temperature *= decay_factor**(1/4)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert state to DataFrame for analysis."""
        return pd.DataFrame({
            'time': [self.time],
            'temperature': [self.temperature],
            'entropy': [self.entropy],
            'information_content': [self.information_content],
            'density_mean': [np.mean(self.density)],
            'density_std': [np.std(self.density)]
        }) 