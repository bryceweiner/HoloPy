"""
Unit tests for Hilbert space management.
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from holopy.core.hilbert import HilbertSpace
from holopy.config.constants import (
    INFORMATION_GENERATION_RATE,
    PLANCK_CONSTANT,
    CRITICAL_THRESHOLD
)

@pytest.fixture
def hilbert_space():
    """Create temporary HilbertSpace instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield HilbertSpace(100, 1.0, Path(tmpdir))

def test_hilbert_space_initialization(hilbert_space):
    """Test Hilbert space initialization with holographic constraints."""
    assert hilbert_space.dimension == 100
    assert hilbert_space.boundary_radius == 1.0
    assert hilbert_space.basis_states.shape == (100, 100)
    
    # Check basis orthonormality
    for i in range(100):
        for j in range(100):
            overlap = np.vdot(hilbert_space.basis_states[i], hilbert_space.basis_states[j])
            if i == j:
                assert np.abs(overlap - 1.0) < 1e-10
            else:
                assert np.abs(overlap) < 1e-10

def test_state_persistence(hilbert_space):
    """Test state persistence with holographic constraints."""
    # Create test state
    state = np.random.normal(0, 1, 100) + 1j * np.random.normal(0, 1, 100)
    state /= np.sqrt(np.vdot(state, state))
    
    # Save state
    path = hilbert_space.save(state, {"test": True}, 0.0)
    assert path.exists()
    
    # Load state
    loaded_state, metadata = hilbert_space.load(version_id=path.stem.replace("state_", ""))
    assert np.allclose(loaded_state, hilbert_space.project_state(state), atol=1e-10)
    assert metadata["test"] is True

def test_entropy_bound_enforcement(hilbert_space):
    """Test enforcement of holographic entropy bounds."""
    # Create high entropy state
    state = np.ones(100, dtype=np.complex128)
    state /= np.sqrt(np.vdot(state, state))
    
    # Project and check entropy
    projected = hilbert_space.project_state(state)
    entropy = hilbert_space.calculate_entropy(projected)
    assert entropy <= hilbert_space.max_information

def test_holographic_basis_properties(hilbert_space):
    """Test properties of holographic basis states."""
    for n in range(hilbert_space.dimension):
        basis_state = hilbert_space.basis_states[n]
        
        # Check normalization
        assert np.abs(np.vdot(basis_state, basis_state) - 1.0) < 1e-10
        
        # Check holographic decay
        k = 2 * np.pi * n / (hilbert_space.boundary_radius * hilbert_space.dimension)
        expected_decay = np.exp(-INFORMATION_GENERATION_RATE * abs(k))
        max_amplitude = np.max(np.abs(basis_state))
        assert np.abs(max_amplitude - expected_decay) < 1e-10

def test_hamiltonian_properties():
    """Test properties of the holographic Hamiltonian."""
    dimension = 100
    boundary_radius = 1.0
    
    hilbert = HilbertSpace(dimension, boundary_radius)
    
    # Check Hermiticity
    ham_dense = hilbert.hamiltonian.toarray()
    assert np.allclose(ham_dense, ham_dense.conj().T)
    
    # Check energy spectrum is bounded
    eigenvalues = np.linalg.eigvalsh(ham_dense)
    assert np.all(eigenvalues >= 0)  # Energy should be positive
    assert np.all(eigenvalues <= hilbert.cutoff_energy)  # Respect UV cutoff 