"""
Unit tests for holographic error mitigation.
"""
import pytest
import numpy as np
from holopy.quantum.error_mitigation import (
    HolographicMitigation,
    MitigationResult
)
from holopy.quantum.noise import HolographicNoise
from holopy.config.constants import INFORMATION_GENERATION_RATE

@pytest.fixture
def setup_mitigation():
    """Setup test error mitigation system."""
    n_qubits = 2
    noise_model = HolographicNoise(n_qubits, 0.1)
    return HolographicMitigation(n_qubits, noise_model)

def test_mitigation_initialization(setup_mitigation):
    """Test error mitigation initialization."""
    mitigation = setup_mitigation
    
    # Verify basic properties
    assert mitigation.n_qubits == 2
    assert mitigation.noise_model is not None
    assert len(mitigation.scale_factors) > 0

def test_error_mitigation(setup_mitigation):
    """Test complete error mitigation process."""
    mitigation = setup_mitigation
    
    # Create test state
    state = np.zeros(2**mitigation.n_qubits, dtype=complex)
    state[0] = 1.0
    
    # Apply noise
    noisy_state = mitigation.noise_model.apply_noise(state, 0.1)
    
    # Apply mitigation
    result = mitigation.mitigate_errors(noisy_state, state)
    
    # Verify mitigation results
    assert isinstance(result, MitigationResult)
    assert result.fidelity_improvement > 0
    assert result.error_reduction > 0
    assert 0 <= result.confidence <= 1

def test_holographic_constraints(setup_mitigation):
    """Test holographic constraints in error mitigation."""
    mitigation = setup_mitigation
    
    # Create superposition state
    state = np.ones(2**mitigation.n_qubits, dtype=complex)
    state /= np.sqrt(len(state))
    
    # Apply noise and mitigation
    noisy_state = mitigation.noise_model.apply_noise(state, 0.1)
    result = mitigation.mitigate_errors(noisy_state)
    
    # Verify entropy bounds
    mitigated_density = np.abs(result.mitigated_state)**2
    entropy = -np.sum(mitigated_density * np.log2(mitigated_density + 1e-10))
    assert entropy <= mitigation.n_qubits

def test_mitigation_scaling(setup_mitigation):
    """Test error mitigation scaling with noise strength."""
    mitigation = setup_mitigation
    
    # Create test state
    state = np.zeros(2**mitigation.n_qubits, dtype=complex)
    state[0] = 1.0
    
    # Test different noise strengths
    times = [0.1, 0.2, 0.3]
    improvements = []
    
    for t in times:
        noisy_state = mitigation.noise_model.apply_noise(state, t)
        result = mitigation.mitigate_errors(noisy_state, state)
        improvements.append(result.fidelity_improvement)
    
    # Verify mitigation effectiveness decreases with noise
    assert all(x >= y for x, y in zip(improvements, improvements[1:])) 