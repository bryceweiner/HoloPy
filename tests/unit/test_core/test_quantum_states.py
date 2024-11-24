"""
Unit tests for quantum state management.
"""
import pytest
import numpy as np
from holopy.core.quantum_states import QuantumState
from holopy.config.constants import INFORMATION_GENERATION_RATE

def test_quantum_state_creation():
    """Test creation of initial quantum state."""
    grid_points = 100
    spatial_extent = 1.0
    
    state = QuantumState.create_initial_state(grid_points, spatial_extent)
    
    assert isinstance(state, QuantumState)
    assert state.wavefunction.shape == (grid_points,)
    assert np.abs(np.sum(np.abs(state.wavefunction)**2) - 1.0) < 1e-10
    assert state.time == 0.0
    assert state.coherence == 1.0

def test_quantum_state_evolution():
    """Test quantum state evolution with holographic corrections."""
    grid_points = 100
    spatial_extent = 1.0
    dt = 1e-6
    
    state = QuantumState.create_initial_state(grid_points, spatial_extent)
    initial_norm = np.sum(np.abs(state.wavefunction)**2)
    
    state.evolve(dt)
    
    # Check holographic decay
    expected_coherence = np.exp(-INFORMATION_GENERATION_RATE * dt)
    assert np.abs(state.coherence - expected_coherence) < 1e-10
    
    # Check norm preservation with holographic correction
    final_norm = np.sum(np.abs(state.wavefunction)**2)
    assert np.abs(final_norm - initial_norm * expected_coherence) < 1e-10

def test_observable_calculation():
    """Test calculation of quantum observables."""
    grid_points = 100
    spatial_extent = 1.0
    
    state = QuantumState.create_initial_state(grid_points, spatial_extent)
    x_expect, p_expect, energy = state.calculate_observables()
    
    # Gaussian wavepacket should be centered at x=0
    assert np.abs(x_expect) < 1e-10
    
    # Ground state energy should be positive
    assert energy > 0 