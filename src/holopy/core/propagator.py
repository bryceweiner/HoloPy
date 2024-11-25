"""
Field propagator module implementing holographic corrections and active inference.
"""
from typing import Optional, Dict, Tuple
import numpy as np
import scipy
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import expm_multiply
import logging
from functools import lru_cache
from ..config.constants import (
    INFORMATION_GENERATION_RATE,
    COUPLING_CONSTANT,
    SPEED_OF_LIGHT,
    PLANCK_CONSTANT,
    E8_DIMENSION,
    CRITICAL_THRESHOLD
)
from scipy.fft import fft, ifft
from ..optimization.state_cache import LRUStateCache
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CacheMetrics:
    """Metrics for cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
@dataclass
class KernelCache:
    """Cache for propagator kernels."""
    max_size: int = 1000
    _cache: Dict[Tuple[float, float], complex] = None
    
    def __post_init__(self):
        """Initialize cache dictionary."""
        self._cache = {}
    
    def get(self, x1: float, x2: float) -> Optional[complex]:
        """Get cached kernel value."""
        return self._cache.get((x1, x2))
    
    def set(self, x1: float, x2: float, value: complex) -> None:
        """Set kernel value in cache."""
        if len(self._cache) >= self.max_size:
            # Remove oldest entry
            self._cache.pop(next(iter(self._cache)))
        self._cache[(x1, x2)] = value

class FieldPropagator:
    """
    Implements the field propagator with holographic corrections and active inference.
    """
    
    def __init__(
        self,
        spatial_points: int,
        dt: float,
        spatial_extent: float,
        cache_size: int = 1000,
        cache_maxbytes: Optional[int] = None
    ):
        """
        Initialize the field propagator.
        
        Args:
            spatial_points: Number of spatial grid points
            dt: Time step
            spatial_extent: Physical size of the simulation domain
            cache_size: Size of the propagator cache
            cache_maxbytes: Maximum size of the cache in bytes
        """
        self.spatial_points = spatial_points
        self.dt = dt
        self.dx = spatial_extent / spatial_points
        
        # Initialize spatial grid and momentum space
        self.x_grid = np.linspace(-spatial_extent/2, spatial_extent/2, spatial_points)
        self.k_grid = 2 * np.pi * np.fft.fftfreq(spatial_points, self.dx)
        
        # Initialize kernel cache
        self.kernel_cache = KernelCache(max_size=cache_size)
        
        # Precompute k-space quantities
        self._initialize_k_space()
        
        logger.info(
            f"Initialized FieldPropagator with {spatial_points} points, "
            f"dx={self.dx:.6f}, dt={dt:.6f}"
        )
    
    def _initialize_k_space(self) -> None:
        """Initialize k-space quantities."""
        try:
            # Calculate k^2 terms
            self.k2 = self.k_grid**2
            
            # Calculate holographic corrections
            self.holo_corrections = self._holographic_correction(self.k2)
            
            # Calculate propagation phase
            self.propagation_phase = -0.5 * self.k2 * self.dt * (1.0 + self.holo_corrections)
            
        except Exception as e:
            logger.error(f"K-space initialization failed: {str(e)}")
            raise
    
    def _holographic_correction(self, k_squared: np.ndarray) -> np.ndarray:
        """
        Calculate holographic correction to kinetic term.
        
        Args:
            k_squared: Square of momentum values
            
        Returns:
            Correction factors for each k value
        """
        try:
            # Avoid division by zero
            k_mag = np.sqrt(np.abs(k_squared))
            epsilon = 1e-10
            
            # Calculate correction with numerical stability
            correction = INFORMATION_GENERATION_RATE * k_mag / (SPEED_OF_LIGHT * (k_mag + epsilon))
            
            return correction
            
        except Exception as e:
            logger.error(f"Holographic correction calculation failed: {str(e)}")
            raise
    
    def get_kernel(self, x1: float, x2: float) -> complex:
        """
        Calculate the propagator kernel between two points.
        
        Args:
            x1: First position
            x2: Second position
            
        Returns:
            Complex kernel value
        """
        try:
            # Check cache
            cached_value = self.kernel_cache.get(x1, x2)
            if cached_value is not None:
                return cached_value
            
            # Calculate spatial separation
            dx = x2 - x1
            
            # Calculate phase factors
            spatial_phase = np.exp(1j * self.k_grid * dx)
            evolution_phase = np.exp(self.propagation_phase)
            
            # Calculate kernel
            kernel = np.sum(spatial_phase * evolution_phase) / self.spatial_points
            
            # Cache result
            self.kernel_cache.set(x1, x2, kernel)
            
            return kernel
            
        except Exception as e:
            logger.error(f"Kernel calculation failed: {str(e)}")
            raise
    
    def propagate(
        self,
        wavefunction: np.ndarray,
        steps: int = 1
    ) -> np.ndarray:
        """
        Propagate wavefunction forward in time.
        
        Args:
            wavefunction: Initial state
            steps: Number of time steps
            
        Returns:
            Evolved wavefunction
        """
        try:
            psi = wavefunction.copy()
            
            for _ in range(steps):
                # Transform to k-space
                psi_k = np.fft.fft(psi)
                
                # Apply evolution
                psi_k *= np.exp(self.propagation_phase)
                
                # Transform back
                psi = np.fft.ifft(psi_k)
                
                # Normalize
                psi /= np.linalg.norm(psi)
            
            return psi
            
        except Exception as e:
            logger.error(f"Propagation failed: {str(e)}")
            raise

from ..config.constants import (
    INFORMATION_GENERATION_RATE,
    COUPLING_CONSTANT,
    SPEED_OF_LIGHT,
    PLANCK_CONSTANT,
    E8_DIMENSION,
    CRITICAL_THRESHOLD
)
import logging

logger = logging.getLogger(__name__)

class DualContinuumPropagator:
    """Manages evolution of coupled quantum-classical states."""
    
    def __init__(
        self,
        spatial_points: int,
        spatial_extent: float,
        dt: float
    ):
        """Initialize propagator with k-space grid."""
        self.spatial_points = spatial_points
        self.spatial_extent = spatial_extent
        self.dt = dt
        
        # Initialize k-space grid
        self.dk = 2 * np.pi / spatial_extent
        self.k_space = self.dk * np.fft.fftfreq(spatial_points) * spatial_points
        
        # Initialize operators
        self.kinetic_operator = self._initialize_kinetic_operator()
        
    def _initialize_kinetic_operator(self) -> np.ndarray:
        """Initialize kinetic energy operator."""
        try:
            # Create Laplacian in k-space (now k_space is defined)
            laplacian_k = -(self.k_space**2)
            return 0.5 * laplacian_k  # Kinetic energy operator
        except Exception as e:
            logger.error(f"Kinetic operator initialization failed: {str(e)}")
            raise

"""
Field propagator implementation with caching support.
"""
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from ..config.constants import (
    INFORMATION_GENERATION_RATE,
    SPEED_OF_LIGHT,
    COUPLING_CONSTANT
)
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheMetrics:
    """Metrics for cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class CachedPropagator:
    """Field propagator with integrated caching."""
    
    def __init__(
        self,
        spatial_points: int,
        spatial_extent: float,
        cache_size: int = 1000
    ):
        """Initialize propagator with caching."""
        self.spatial_points = spatial_points
        self.spatial_extent = spatial_extent
        self.dx = spatial_extent / spatial_points
        self.k_space = 2 * np.pi * np.fft.fftfreq(spatial_points, self.dx)
        
        # Initialize cache and metrics
        self.cache: Dict[Tuple[float, float], np.ndarray] = {}
        self.cache_size = cache_size
        self.metrics = CacheMetrics()
        
        logger.info(
            f"Initialized CachedPropagator with {spatial_points} points, "
            f"cache_size={cache_size}"
        )
    
    def get_propagator(
        self,
        dt: float,
        gamma: float = INFORMATION_GENERATION_RATE
    ) -> np.ndarray:
        """Get cached propagator or compute if not in cache."""
        cache_key = (dt, gamma)
        
        # Check cache
        if cache_key in self.cache:
            self.metrics.hits += 1
            return self.cache[cache_key]
            
        self.metrics.misses += 1
        
        # Compute propagator
        k = self.k_space
        propagator = np.exp(-1j * k**2 * dt / 2)
        propagator *= np.exp(-gamma * dt) * (1 + gamma * np.abs(k) / SPEED_OF_LIGHT)
        
        # Cache management
        if len(self.cache) >= self.cache_size:
            self.metrics.evictions += 1
            self.cache.pop(next(iter(self.cache)))
        
        # Update metrics
        self.metrics.total_bytes += propagator.nbytes
        self.cache[cache_key] = propagator
        
        return propagator
    def propagate(
        self,
        wavefunction: np.ndarray,
        dt: float,
        gamma: Optional[float] = None
    ) -> np.ndarray:
        """
        Propagate quantum state using cached propagator.
        
        Args:
            wavefunction: Initial quantum state
            dt: Time step
            gamma: Optional custom information generation rate
            
        Returns:
            Evolved quantum state
        """
        try:
            # Get propagator (cached or computed)
            propagator = self.get_propagator(dt, gamma or INFORMATION_GENERATION_RATE)
            
            # Transform to k-space
            k_space = np.fft.fft(wavefunction)
            
            # Apply propagator
            evolved = k_space * propagator
            
            # Transform back to real space
            result = np.fft.ifft(evolved)
            
            # Normalize if needed
            norm = np.linalg.norm(result)
            if abs(norm - 1.0) > 1e-10:
                result = result / norm
                
            return result
            
        except Exception as e:
            logger.error(f"Propagation failed: {str(e)}")
            raise      
    def get_metrics(self) -> Dict[str, float]:
        """Get cache performance metrics."""
        return {
            "hits": self.metrics.hits,
            "misses": self.metrics.misses,
            "evictions": self.metrics.evictions,
            "hit_rate": self.metrics.hit_rate,
            "total_bytes": self.metrics.total_bytes,
            "cache_size": len(self.cache),
            "max_size": self.cache_size
        }

class QuantumPropagator:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.system_size = 2**n_qubits
        
    def propagate_state(
        self,
        state: np.ndarray,
        hamiltonian: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Propagate quantum state."""
        try:
            evolution = scipy.linalg.expm(-1j * hamiltonian * dt)
            return evolution @ state
        except Exception as e:
            logger.error(f"State propagation failed: {str(e)}")
            raise
