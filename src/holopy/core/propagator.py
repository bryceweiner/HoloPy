"""
Field propagator module implementing holographic corrections and active inference.
"""
from typing import Optional, Dict, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
import logging
from functools import lru_cache
from ..config.constants import (
    INFORMATION_GENERATION_RATE,
    COUPLING_CONSTANT,
    SPEED_OF_LIGHT
)

logger = logging.getLogger(__name__)

class FieldPropagator:
    """
    Implements the field propagator with holographic corrections and active inference.
    """
    
    def __init__(
        self,
        spatial_points: int,
        dt: float,
        spatial_extent: float,
        cache_size: int = 1000
    ):
        """
        Initialize the field propagator.
        
        Args:
            spatial_points: Number of spatial grid points
            dt: Time step
            spatial_extent: Physical size of the simulation domain
            cache_size: Size of the propagator cache
        """
        self.spatial_points = spatial_points
        self.dt = dt
        self.dx = spatial_extent / spatial_points
        self.cache_size = cache_size
        
        # Initialize spatial grid and momentum space
        self.x_grid = np.linspace(-spatial_extent/2, spatial_extent/2, spatial_points)
        self.k_grid = 2 * np.pi * np.fft.fftfreq(spatial_points, self.dx)
        
        # Initialize propagator components
        self._initialize_propagator()
        
        logger.info(
            f"Initialized FieldPropagator with {spatial_points} points, "
            f"dt={dt:.2e}, dx={self.dx:.2e}"
        )
    
    def _initialize_propagator(self) -> None:
        """Initialize the propagator components."""
        # Construct kinetic and potential terms
        self.kinetic = self._construct_kinetic()
        self.potential = self._construct_potential()
        
        # Initialize caches
        self._propagator_cache: Dict[float, csr_matrix] = {}
        self._kernel_cache: Dict[Tuple[float, float], float] = {}
    
    def _construct_kinetic(self) -> csr_matrix:
        """Construct the kinetic energy operator."""
        # Include holographic corrections in momentum space
        k2 = self.k_grid**2
        kinetic_diag = -0.5 * k2 * (1 + self._holographic_correction(k2))
        return csr_matrix(np.diag(kinetic_diag))
    
    def _construct_potential(self) -> csr_matrix:
        """Construct the potential energy operator."""
        # Include coupling terms and active inference
        potential_diag = (
            COUPLING_CONSTANT * self.x_grid**2 
            + self._active_inference_potential()
        )
        return csr_matrix(np.diag(potential_diag))
    
    @lru_cache(maxsize=1000)
    def _holographic_correction(self, k2: float) -> float:
        """Calculate holographic corrections to the propagator."""
        # Implement corrections based on holographic principle
        return (
            INFORMATION_GENERATION_RATE 
            * np.log(1 + k2 / (SPEED_OF_LIGHT**2))
            / (4 * np.pi)
        )
    
    def _active_inference_potential(self) -> np.ndarray:
        """Calculate the active inference contribution to the potential."""
        # Implement active inference through an effective potential
        return (
            INFORMATION_GENERATION_RATE 
            * np.log(1 + np.abs(self.x_grid))
            / (2 * np.pi)
        )
    
    def _get_propagator(self, time: float) -> csr_matrix:
        """Get or compute the propagator for a given time step."""
        if time not in self._propagator_cache:
            # Compute new propagator
            hamiltonian = self.kinetic + self.potential
            propagator = expm_multiply(
                -1j * time * hamiltonian,
                np.eye(self.spatial_points)
            )
            
            # Cache management
            if len(self._propagator_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_time = min(self._propagator_cache.keys())
                del self._propagator_cache[oldest_time]
            
            self._propagator_cache[time] = propagator
            
        return self._propagator_cache[time]
    
    def propagate(
        self,
        state: np.ndarray,
        time: float,
        include_corrections: bool = True
    ) -> np.ndarray:
        """
        Propagate a state forward in time.
        
        Args:
            state: Initial state vector
            time: Propagation time
            include_corrections: Whether to include holographic corrections
            
        Returns:
            Propagated state vector
        """
        if state.shape != (self.spatial_points,):
            raise ValueError(
                f"State shape {state.shape} does not match "
                f"spatial points {self.spatial_points}"
            )
        
        propagator = self._get_propagator(time)
        propagated_state = propagator @ state
        
        if include_corrections:
            # Apply holographic and active inference corrections
            propagated_state *= np.exp(
                -INFORMATION_GENERATION_RATE * time / 2
            )
            
        return propagated_state
    
    def get_kernel(self, x1: float, x2: float) -> complex:
        """Calculate the propagator kernel between two points."""
        key = (x1, x2)
        if key not in self._kernel_cache:
            # Calculate kernel with holographic corrections
            dx = x2 - x1
            k2 = self.k_grid**2
            phase = np.exp(1j * self.k_grid * dx)
            
            kernel = np.sum(
                phase * np.exp(-0.5 * k2 * self.dt * (1 + self._holographic_correction(k2)))
            ) / self.spatial_points
            
            # Cache management
            if len(self._kernel_cache) >= self.cache_size:
                # Remove a random entry (simple cache management)
                del self._kernel_cache[next(iter(self._kernel_cache))]
                
            self._kernel_cache[key] = kernel
            
        return self._kernel_cache[key]