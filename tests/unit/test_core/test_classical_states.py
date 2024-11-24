"""
Unit tests for classical observable management.
"""
import pytest
import numpy as np
from holopy.core.classical_states import ContinuumState
from holopy.config.constants import (
    INFORMATION_GENERATION_RATE,
    BOLTZMANN_CONSTANT,
    PLANCK_CONSTANT
)

def test_classical_state_creation():
    """Test creation of classical state from quantum state."""
    # Create test quantum state (Gaussian)
    grid_points = 100
    x = np.linspace(-1, 1, grid_points)
    psi = np.exp(-x**2 / (2 * 0.1**2))
    psi /= np.sqrt(np.sum(np.abs(psi)**2))
    
    state = ContinuumState.from_quantum_state(psi, 0.0, x)
    
    assert isinstance(state, ContinuumState)
    assert state.time == 0.0
    assert state.temperature > 0
    assert state.entropy > 0
    assert state.information_content > 0
    assert np.allclose(np.sum(state.density), 1.0)

def test_classical_state_evolution():
    """Test classical state evolution with holographic corrections."""
    grid_points = 100
    x = np.linspace(-1, 1, grid_points)
    psi = np.exp(-x**2 / (2 * 0.1**2))
    psi /= np.sqrt(np.sum(np.abs(psi)**2))
    
    state = ContinuumState.from_quantum_state(psi, 0.0, x)
    
    initial_temp = state.temperature
    initial_entropy = state.entropy
    
    dt = 1e-6
    state.evolve(dt)
    
    # Check temperature evolution
    expected_temp = initial_temp * np.exp(-INFORMATION_GENERATION_RATE * dt/4)
    assert np.abs(state.temperature - expected_temp) < 1e-10
    
    # Check entropy evolution
    expected_entropy = initial_entropy * np.exp(-INFORMATION_GENERATION_RATE * dt)
    assert np.abs(state.entropy - expected_entropy) < 1e-10 