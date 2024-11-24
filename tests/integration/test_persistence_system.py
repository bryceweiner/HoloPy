"""
Integration tests for persistence system with HilbertSpace.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
from holopy.core.hilbert import HilbertSpace
from holopy.core.quantum_states import QuantumState
from holopy.metrics.validation_suite import HolographicValidationSuite

@pytest.fixture
def setup_system():
    """Set up complete test system."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hilbert = HilbertSpace(100, 1.0, Path(tmpdir))
        validator = HolographicValidationSuite()
        yield hilbert, validator

def test_complete_persistence_cycle(setup_system):
    """Test complete persistence cycle with HilbertSpace integration."""
    hilbert, validator = setup_system
    
    # Create and evolve quantum state
    state = QuantumState.create_initial_state(100, 1.0)
    times = np.linspace(0, 0.1, 10)
    
    for t in times:
        # Save state through HilbertSpace
        path = hilbert.save(
            state.wavefunction,
            {"time": t},
            t,
            is_checkpoint=(t == times[-1])
        )
        
        # Load and validate
        loaded_state, metadata = hilbert.load(
            version_id=path.stem.replace("state_", "")
        )
        
        # Verify holographic constraints
        assert hilbert.calculate_entropy(loaded_state) <= hilbert.max_information
        assert np.abs(np.vdot(loaded_state, loaded_state) - 1.0) < 1e-10
        
        # Evolve state
        state.evolve(times[1] - times[0])
        state.wavefunction = hilbert.project_state(state.wavefunction)

def test_checkpoint_recovery(setup_system):
    """Test checkpoint recovery through HilbertSpace."""
    hilbert, _ = setup_system
    
    # Create checkpoints
    state = QuantumState.create_initial_state(100, 1.0)
    for t in [0.0, 0.1, 0.2]:
        hilbert.save(state.wavefunction, {"time": t}, t, is_checkpoint=True)
        state.evolve(0.1)
    
    # Load latest checkpoint
    loaded_state, metadata = hilbert.load(latest_checkpoint=True)
    assert metadata["time"] == 0.2
    assert hilbert.calculate_entropy(loaded_state) <= hilbert.max_information 