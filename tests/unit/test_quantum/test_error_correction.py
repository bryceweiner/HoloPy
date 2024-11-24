"""
Unit tests for holographic quantum error correction.
"""
import pytest
import numpy as np
from holopy.quantum.error_correction import (
    HolographicStabilizer,
    StabilizerMetrics
)
from holopy.config.constants import INFORMATION_GENERATION_RATE

@pytest.fixture
def setup_stabilizer():
    """Setup test stabilizer system."""
    code_distance = 3
    spatial_points = 64
    return HolographicStabilizer(code_distance, spatial_points)

def test_stabilizer_initialization(setup_stabilizer):
    """Test stabilizer code initialization."""
    stabilizer = setup_stabilizer
    
    # Verify stabilizer structure
    assert len(stabilizer.stabilizers) == stabilizer.code_distance**2 - 1
    assert stabilizer.logical_x is not None
    assert stabilizer.logical_z is not None
    
    # Check holographic constraints
    for op in stabilizer.stabilizers:
        weight = np.sum(np.abs(op.data))
        assert weight <= np.log2(stabilizer.spatial_points)

def test_syndrome_measurement(setup_stabilizer):
    """Test syndrome measurement with holographic noise."""
    stabilizer = setup_stabilizer
    
    # Create test state with known error
    state = np.zeros(stabilizer.spatial_points, dtype=complex)
    state[0] = 1.0
    
    # Apply test error
    error_pos = stabilizer.spatial_points // 2
    state[error_pos] = 0.1
    state /= np.linalg.norm(state)
    
    # Measure syndrome
    syndrome, metrics = stabilizer.measure_syndrome(state)
    
    # Verify syndrome properties
    assert isinstance(syndrome, np.ndarray)
    assert syndrome.dtype == bool
    assert len(syndrome) == len(stabilizer.stabilizers)
    
    # Check metrics
    assert isinstance(metrics, StabilizerMetrics)
    assert 0 <= metrics.correction_fidelity <= 1
    assert metrics.code_distance == stabilizer.code_distance

def test_error_correction(setup_stabilizer):
    """Test error correction with holographic constraints."""
    stabilizer = setup_stabilizer
    
    # Create test state
    state = np.zeros(stabilizer.spatial_points, dtype=complex)
    state[0] = 1.0
    
    # Apply known error
    error_state = state.copy()
    error_state[1] = 0.1
    error_state /= np.linalg.norm(error_state)
    
    # Measure and correct
    syndrome, _ = stabilizer.measure_syndrome(error_state)
    corrected_state, fidelity = stabilizer.apply_correction(error_state, syndrome)
    
    # Verify correction
    assert np.abs(np.vdot(corrected_state, state)) > 0.9
    assert 0 <= fidelity <= 1

def test_holographic_bounds(setup_stabilizer):
    """Test holographic bounds on error correction."""
    stabilizer = setup_stabilizer
    
    # Create maximally mixed state
    state = np.ones(stabilizer.spatial_points, dtype=complex)
    state /= np.linalg.norm(state)
    
    # Measure syndrome
    syndrome, metrics = stabilizer.measure_syndrome(state)
    
    # Verify holographic entropy bound
    entropy = -np.sum(np.abs(state)**2 * np.log(np.abs(state)**2 + 1e-10))
    assert entropy <= stabilizer.spatial_points
    
    # Check information rate constraints
    assert metrics.logical_error_rate >= INFORMATION_GENERATION_RATE 