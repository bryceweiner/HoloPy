"""
Unit tests for holographic measurement system.
"""
import pytest
import numpy as np
from holopy.quantum.measurement import (
    HolographicMeasurement,
    MeasurementResult,
    TomographyResult
)
from holopy.quantum.error_correction import HolographicStabilizer
from holopy.config.constants import INFORMATION_GENERATION_RATE

@pytest.fixture
def setup_measurement():
    """Setup test measurement system."""
    n_qubits = 3
    bases = ['X', 'Y', 'Z']
    stabilizer = HolographicStabilizer(3, 8)
    return HolographicMeasurement(n_qubits, bases, stabilizer)

def test_measurement_initialization(setup_measurement):
    """Test measurement system initialization."""
    measurement = setup_measurement
    
    # Verify operator initialization
    assert all(basis in measurement.operators for basis in ['X', 'Y', 'Z'])
    assert measurement.n_qubits == 3
    
    # Check operator properties
    for op in measurement.operators.values():
        assert np.allclose(op @ op, np.eye(2))

def test_single_qubit_measurement(setup_measurement):
    """Test single qubit measurement with holographic constraints."""
    measurement = setup_measurement
    
    # Create test state (|0⟩ state)
    state = np.zeros(2**measurement.n_qubits, dtype=complex)
    state[0] = 1.0
    
    # Measure in Z basis
    result = measurement.measure_state(state, 'Z', 0)
    
    # Verify result properties
    assert isinstance(result, MeasurementResult)
    assert -1 <= result.expectation_value <= 1
    assert result.uncertainty >= 0
    assert 0 <= result.collapse_fidelity <= 1
    assert result.information_gain >= 0

def test_measurement_uncertainty(setup_measurement):
    """Test measurement uncertainty scaling."""
    measurement = setup_measurement
    
    # Create superposition state
    state = np.ones(2**measurement.n_qubits, dtype=complex)
    state /= np.sqrt(len(state))
    
    # Measure multiple times
    uncertainties = []
    for _ in range(100):
        result = measurement.measure_state(state, 'X', 0)
        uncertainties.append(result.uncertainty)
    
    # Verify uncertainty properties
    mean_uncertainty = np.mean(uncertainties)
    assert mean_uncertainty >= INFORMATION_GENERATION_RATE
    assert np.std(uncertainties) > 0

def test_quantum_tomography(setup_measurement):
    """Test quantum state tomography with holographic constraints."""
    measurement = setup_measurement
    
    # Create test state (|+⟩ state on first qubit)
    state = np.zeros(2**measurement.n_qubits, dtype=complex)
    state[0] = 1/np.sqrt(2)
    state[1] = 1/np.sqrt(2)
    
    # Perform tomography
    result = measurement.perform_tomography(state, n_measurements=100)
    
    # Verify tomography results
    assert isinstance(result, TomographyResult)
    assert result.density_matrix.shape == (2**measurement.n_qubits,)*2
    assert result.fidelity > 0.8  # Allow for some reconstruction error
    assert 0 <= result.purity <= 1
    assert result.entropy >= 0
    assert 0 <= result.confidence <= 1 