import pytest
import numpy as np
from holopy.core.hilbert_continuum import DualState

def test_dual_state_initialization(dual_state):
    assert isinstance(dual_state, DualState)
    assert isinstance(dual_state.quantum_state, np.ndarray)
    assert isinstance(dual_state.classical_density, np.ndarray)
    assert dual_state.time == 0.0
    assert isinstance(dual_state.coherence_hierarchy, tuple)

def test_create_initial_state(hilbert_continuum):
    hilbert_continuum.create_initial_state()
    assert hasattr(hilbert_continuum, "matter_wavefunction")
    assert hasattr(hilbert_continuum, "antimatter_wavefunction")
    assert np.abs(
        np.sum(np.abs(hilbert_continuum.matter_wavefunction)**2) - 1.0
    ) < 1e-6

def test_calculate_temperature(hilbert_continuum):
    hilbert_continuum.create_initial_state()
    temp = hilbert_continuum._calculate_temperature()
    assert isinstance(temp, float)
    assert temp >= 0
    assert np.isfinite(temp)

def test_calculate_coupling_strength(hilbert_continuum):
    hilbert_continuum.create_initial_state()
    coupling = hilbert_continuum._calculate_coupling_strength()
    assert isinstance(coupling, float)
    assert 0 <= coupling <= 1
    assert np.isfinite(coupling)

def test_calculate_entanglement(hilbert_continuum):
    hilbert_continuum.create_initial_state()
    entanglement = hilbert_continuum._calculate_entanglement()
    
    # Basic validation of entanglement value
    assert isinstance(entanglement, float)
    assert entanglement >= 0
    assert np.isfinite(entanglement)
    
    # For initial state (identical matter/antimatter states),
    # entanglement should be minimal
    assert entanglement < 1e-6

def test_entanglement_with_different_states(hilbert_continuum):
    hilbert_continuum.create_initial_state()
    
    # Create a significantly different antimatter state
    hilbert_continuum.antimatter_wavefunction = np.roll(
        hilbert_continuum.antimatter_wavefunction,
        shift=hilbert_continuum.dimension // 2
    )
    
    entanglement = hilbert_continuum._calculate_entanglement()
    assert entanglement >= 0  # Must be non-negative
    assert entanglement > 1e-6  # Should have measurable entanglement