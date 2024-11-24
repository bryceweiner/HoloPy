"""
Quantum state management module for the holographic framework.
"""
"""
Quantum state management for dual continuum system.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from .hilbert import HilbertSpace
import logging
logger = logging.getLogger(__name__)
from ..config.constants import (
    INFORMATION_GENERATION_RATE
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


@dataclass
class DualContinuumState:
    """Manages quantum states for matter-antimatter continuum system."""
    
    # Wavefunctions
    matter_wavefunction: np.ndarray
    antimatter_wavefunction: np.ndarray
    
    # Spatial configuration
    spatial_points: int
    spatial_extent: float
    
    # Time tracking
    time: float = 0.0
    
    # Class constants
    INFORMATION_GENERATION_RATE = INFORMATION_GENERATION_RATE
    
    @classmethod
    def create_initial_state(
        cls,
        spatial_points: int,
        spatial_extent: float
    ) -> 'DualContinuumState':
        """Create initial dual continuum state."""
        try:
            # Create spatial grid
            x = np.linspace(-spatial_extent/2, spatial_extent/2, spatial_points)
            
            # Create initial Gaussian state
            sigma = spatial_extent/20
            psi = np.exp(-x**2/(2*sigma**2))
            psi /= np.sqrt(np.sum(np.abs(psi)**2))
            
            logger.info("Created initial dual continuum state")
            
            return cls(
                matter_wavefunction=psi.copy(),
                antimatter_wavefunction=psi.copy(),
                spatial_points=spatial_points,
                spatial_extent=spatial_extent
            )
            
        except Exception as e:
            logger.error(f"Failed to create initial state: {str(e)}")
            raise
    
    def evolve(self, dt: float) -> None:
        """Evolve state forward in time."""
        try:
            # Update time
            self.time += dt
            
            # Apply information rate decay to matter continuum only
            self.matter_wavefunction *= np.exp(-self.INFORMATION_GENERATION_RATE * dt / 2)
            
            # Renormalize states
            self.matter_wavefunction /= np.sqrt(np.sum(np.abs(self.matter_wavefunction)**2))
            self.antimatter_wavefunction /= np.sqrt(np.sum(np.abs(self.antimatter_wavefunction)**2))
            
            logger.debug(f"Evolved state to t={self.time:.6f}")
            
        except Exception as e:
            logger.error(f"Evolution failed: {str(e)}")
            raise
    
    def get_coherence(self) -> float:
        """Calculate cross-continuum coherence."""
        try:
            overlap = np.abs(np.sum(
                np.conj(self.matter_wavefunction) * 
                self.antimatter_wavefunction
            ))
            return overlap / self.spatial_points
            
        except Exception as e:
            logger.error(f"Coherence calculation failed: {str(e)}")
            raise
    
    def to_classical_observables(self) -> pd.DataFrame:
        """Convert quantum state to classical observables."""
        try:
            observables = {
                'time': self.time,
                'matter_density': np.sum(np.abs(self.matter_wavefunction)**2),
                'antimatter_density': np.sum(np.abs(self.antimatter_wavefunction)**2),
                'coherence': self.get_coherence(),
                'matter_energy': np.sum(np.abs(np.fft.fft(self.matter_wavefunction))**2),
                'antimatter_energy': np.sum(np.abs(np.fft.fft(self.antimatter_wavefunction))**2),
                'information_content': -np.sum(
                    np.abs(self.matter_wavefunction)**2 * 
                    np.log2(np.abs(self.matter_wavefunction)**2 + 1e-10)
                )
            }
            
            return pd.DataFrame([observables])
            
        except Exception as e:
            logger.error(f"Observable calculation failed: {str(e)}")
            raise
    
    def project_onto_hilbert_space(self, hilbert: HilbertSpace) -> None:
        """Project states onto holographic Hilbert space."""
        try:
            self.matter_wavefunction = hilbert.project_state(self.matter_wavefunction)
            self.antimatter_wavefunction = hilbert.project_state(self.antimatter_wavefunction)
            
            logger.debug("Projected states onto Hilbert space")
            
        except Exception as e:
            logger.error(f"Hilbert space projection failed: {str(e)}")
            raise