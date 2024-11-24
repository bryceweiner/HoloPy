"""
Unit tests for field propagator.
"""
import pytest
import numpy as np
from holopy.core.propagator import FieldPropagator
from holopy.config.constants import (
    INFORMATION_GENERATION_RATE,
    SPEED_OF_LIGHT
)

def test_propagator_initialization():
    """Test field propagator initialization."""
    spatial_points = 100
    dt = 1e-6
    spatial_extent = 1.0
    
    prop = FieldPropagator(spatial_points, dt, spatial_extent)
    
    assert prop.spatial_points == spatial_points
    assert prop.dt == dt
    assert len(prop.x_grid) == spatial_points
    assert len(prop.k_grid) == spatial_points

def test_state_propagation():
    """Test quantum state propagation with holographic corrections."""
    spatial_points = 100
    dt = 1e-6
    spatial_extent = 1.0
    
    prop = FieldPropagator(spatial_points, dt, spatial_extent)
    
    # Create test state (Gaussian)
    x = prop.x_grid
    state = np.exp(-x**2 / (2 * 0.1**2))
    state /= np.sqrt(np.sum(np.abs(state)**2))
    
    # Propagate state
    evolved_state = prop.propagate(state, dt)
    
    # Check norm with holographic decay
    expected_norm = np.exp(-INFORMATION_GENERATION_RATE * dt)
    actual_norm = np.sqrt(np.sum(np.abs(evolved_state)**2))
    assert np.abs(actual_norm - expected_norm) < 1e-10

def test_kernel_properties():
    """Test properties of the propagator kernel."""
    spatial_points = 100
    dt = 1e-6
    spatial_extent = 1.0
    
    prop = FieldPropagator(spatial_points, dt, spatial_extent)
    
    # Test kernel symmetry
    x1, x2 = 0.1, 0.2
    kernel_12 = prop.get_kernel(x1, x2)
    kernel_21 = prop.get_kernel(x2, x1)
    
    assert np.abs(kernel_12 - kernel_21.conj()) < 1e-10
    
    # Test causality
    dx = x2 - x1
    assert np.abs(kernel_12) <= np.exp(-INFORMATION_GENERATION_RATE * np.abs(dx) / SPEED_OF_LIGHT) 