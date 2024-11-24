"""
Unit tests for holographic quantum channels.
"""
import pytest
import numpy as np
from holopy.quantum.channels import (
    HolographicChannel,
    ChannelMetrics
)
from holopy.config.constants import (
    INFORMATION_GENERATION_RATE,
    PLANCK_CONSTANT
)

@pytest.fixture
def setup_channel():
    """Setup test quantum channel."""
    n_qubits = 2
    temperature = 0.1
    return HolographicChannel(n_qubits, temperature)

def test_channel_initialization(setup_channel):
    """Test quantum channel initialization."""
    channel = setup_channel
    
    # Verify basic properties
    assert channel.n_qubits == 2
    assert channel.temperature > 0
    assert channel.dephasing_rate > 0
    assert len(channel.lindblad_operators) > 0
    
    # Check Lindblad operator properties
    for op in channel.lindblad_operators:
        assert op.shape == (4, 4)  # 2^2 x 2^2 for 2 qubits

def test_state_evolution(setup_channel):
    """Test quantum state evolution through channel."""
    channel = setup_channel
    
    # Create test state (|00⟩ state)
    state = np.zeros(2**channel.n_qubits, dtype=complex)
    state[0] = 1.0
    
    # Evolve state
    time = 0.1
    evolved_state, metrics = channel.evolve_state(state, time)
    
    # Verify evolution properties
    assert isinstance(metrics, ChannelMetrics)
    assert np.allclose(np.sum(np.abs(evolved_state)**2), 1.0)
    assert metrics.channel_fidelity <= 1.0
    assert metrics.decoherence_rate >= INFORMATION_GENERATION_RATE

def test_thermal_effects(setup_channel):
    """Test thermal effects in quantum channel."""
    channel = setup_channel
    
    # Create excited state (|11⟩)
    state = np.zeros(2**channel.n_qubits, dtype=complex)
    state[-1] = 1.0
    
    # Evolve for thermal relaxation
    time = 1.0
    evolved_state, metrics = channel.evolve_state(state, time)
    
    # Check thermal relaxation
    population = np.abs(evolved_state[-1])**2
    thermal_ratio = np.exp(-PLANCK_CONSTANT / channel.temperature)
    assert population < 1.0
    assert metrics.entropy_production > 0

def test_holographic_constraints(setup_channel):
    """Test holographic constraints in channel evolution."""
    channel = setup_channel
    
    # Create superposition state
    state = np.ones(2**channel.n_qubits, dtype=complex)
    state /= np.sqrt(len(state))
    
    # Evolve with different times
    times = [0.1, 0.5, 1.0]
    metrics_list = []
    
    for t in times:
        _, metrics = channel.evolve_state(state, t)
        metrics_list.append(metrics)
    
    # Check information loss scaling
    info_losses = [m.information_loss for m in metrics_list]
    assert all(x <= y for x, y in zip(info_losses, info_losses[1:]))
    
    # Verify entropy bounds
    for metrics in metrics_list:
        assert metrics.entropy_production <= channel.n_qubits * np.log(2) 