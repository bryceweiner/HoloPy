import pytest
import numpy as np
from pathlib import Path
from holopy.metrics.validation_suite import ValidationResult, HolographicValidationSuite
from holopy.core.hilbert import HilbertSpace
from holopy.core.hilbert_continuum import HilbertContinuum, DualState
from unittest.mock import Mock

@pytest.fixture
def validation_suite():
    return HolographicValidationSuite(tolerance=1e-6)

@pytest.fixture
def hilbert_space():
    space = HilbertSpace(dimension=128, extent=10.0)
    if not hasattr(space, 'project_state'):
        space.project_state = lambda x: x
    if not hasattr(space, 'calculate_entropy'):
        space.calculate_entropy = lambda x: -np.sum(np.abs(x)**2 * np.log2(np.abs(x)**2 + 1e-10))
    if not hasattr(space, 'calculate_energy'):
        space.calculate_energy = lambda x: np.sum(np.abs(np.fft.fft(x))**2)
    return space

@pytest.fixture
def mock_visualizer():
    return Mock()

@pytest.fixture
def hilbert_continuum(hilbert_space, mock_visualizer, monkeypatch):
    # Patch the HolographicVisualizer to return our mock
    monkeypatch.setattr(
        "holopy.core.hilbert_continuum.HolographicVisualizer",
        lambda **kwargs: mock_visualizer
    )
    
    return HilbertContinuum(
        hilbert_space=hilbert_space,
        dt=0.01,
        enable_hierarchy=True
    )

@pytest.fixture
def sample_state():
    # Create a normalized test state
    state = np.zeros(128, dtype=complex)
    state[64] = 1.0  # Single peak at center
    return state

@pytest.fixture
def dual_state(sample_state):
    return DualState(
        quantum_state=sample_state,
        classical_density=np.abs(sample_state)**2,
        time=0.0,
        coupling_strength=0.1,
        coherence_hierarchy=(1.0, 0.9, 0.8),
        information_content=1.0
    ) 