"""
Settings module for configuring the holographic simulation parameters.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class SimulationSettings:
    """Configuration settings for the holographic simulation."""
    
    # Simulation Parameters
    dt: float = 1e-6  # Time step in seconds
    max_time: float = 1.0  # Maximum simulation time
    spatial_resolution: float = 1e-9  # Spatial grid resolution
    
    # Numerical Parameters
    integration_method: str = "rk4"  # Integration method (rk4, euler, etc.)
    cache_size: int = 1000  # Maximum size of propagator cache
    tolerance: float = 1e-6  # Numerical tolerance for calculations
    
    # Physical Parameters
    temperature: float = 300.0  # System temperature in Kelvin
    initial_entropy: Optional[float] = None  # Initial entropy (if None, will be calculated)
    
    def validate(self) -> None:
        """Validate settings and enforce constraints."""
        if self.dt <= 0:
            raise ValueError("Time step must be positive")
        if self.spatial_resolution <= 0:
            raise ValueError("Spatial resolution must be positive")
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if self.cache_size < 0:
            raise ValueError("Cache size must be non-negative") 