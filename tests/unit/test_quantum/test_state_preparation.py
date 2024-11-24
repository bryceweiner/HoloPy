"""
Unit tests for holographic state preparation.
"""
import pytest
import numpy as np
from holopy.quantum.state_preparation import (
    HolographicPreparation,
    PreparationMetrics
)
from holopy.quantum.error_mitigation import HolographicMitigation
from holopy.config.constants import INFORMATION_GENERATION_RATE

@pytest.fixture
def setup_preparation():
    """Setup test state preparation system."""
    n_qubits = 3
    error_mitigation = HolographicMitigation(n_qubits)
    return HolographicPreparation(n_qubits, error_mitigation)

def test_preparation_initialization(setup_preparation):
    """Test state preparation initialization."""
    preparation = setup_preparation
    
    # Verify basic properties
    assert preparation.n_qubits == 3
    assert preparation.error_mitigation is not None
    assert len(preparation.basis_states) == 2**3

def test_basic_state_preparation(setup_preparation):
    """Test preparation of basic quantum states."""
    preparation = setup_preparation
    
    # Test |0⟩ state preparation
    target_state = np.zeros(2**preparation.n_qubits, dtype=complex)
    target_state[0] = 1.0
    
    prepared_state, metrics = preparation.prepare_state(target_state)
    
    # Verify preparation quality
    assert isinstance(metrics, PreparationMetrics)
    assert metrics.preparation_fidelity > 0.95
    assert metrics.state_purity > 0.95
    assert metrics.verification_confidence > 0.9

def test_entangled_state_preparation(setup_preparation):
    """Test preparation of entangled states."""
    preparation = setup_preparation
    
    # Prepare GHZ state
    prepared_state, metrics = preparation.prepare_entangled_state("GHZ")
    
    # Verify entanglement properties
    assert 0.9 < metrics.entanglement_entropy <= preparation.n_qubits
    assert metrics.preparation_fidelity > 0.9

def test_preparation_robustness(setup_preparation):
    """Test robustness of state preparation."""
    preparation = setup_preparation
    
    # Create random target state
    target_state = np.random.normal(0, 1, 2**preparation.n_qubits) + \
                  1j * np.random.normal(0, 1, 2**preparation.n_qubits)
    target_state /= np.linalg.norm(target_state)
    
    # Multiple preparation attempts
    fidelities = []
    for _ in range(5):
        prepared_state, metrics = preparation.prepare_state(target_state)
        fidelities.append(metrics.preparation_fidelity)
    
    # Verify consistency
    assert np.std(fidelities) < 0.1
    assert np.mean(fidelities) > 0.8

def test_holographic_constraints(setup_preparation):
    """Test holographic constraints in state preparation."""
    preparation = setup_preparation
    
    # Prepare maximally entangled state
    prepared_state, metrics = preparation.prepare_entangled_state("MAX")
    
    # Verify entropy bounds
    assert metrics.entanglement_entropy <= preparation.n_qubits
    
    # Verify information bounds
    information_content = -np.sum(
        np.abs(prepared_state)**2 * 
        np.log2(np.abs(prepared_state)**2 + 1e-10)
    )
    assert information_content <= preparation.n_qubits