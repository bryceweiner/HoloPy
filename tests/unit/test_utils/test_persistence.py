"""
Unit tests for state persistence system.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import h5py
from holopy.utils.persistence import StatePersistence, CompressionMethod
from holopy.config.constants import INFORMATION_GENERATION_RATE

@pytest.fixture
def persistence():
    """Create temporary persistence system."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield StatePersistence(
            Path(tmpdir),
            compression_method=CompressionMethod.BLOSC,
            compression_level=6
        )

def test_state_save_load(persistence):
    """Test basic state save and load functionality."""
    # Create test state
    state = np.random.random(100) + 1j * np.random.random(100)
    state /= np.sqrt(np.sum(np.abs(state)**2))
    metadata = {"time": 0.0, "test_param": 1.0}
    
    # Save state
    state_path = persistence.save_state(state, metadata, 0.0)
    assert state_path.exists()
    
    # Load state
    loaded_state, loaded_metadata = persistence.load_state(
        version_id=state_path.stem.replace("state_", "")
    )
    
    assert np.allclose(loaded_state, state, atol=1e-10)
    assert loaded_metadata["test_param"] == metadata["test_param"]

def test_compression_methods(persistence):
    """Test different compression methods."""
    state = np.random.random(1000) + 1j * np.random.random(1000)
    
    for method in CompressionMethod:
        persistence.compression_method = method
        compressed, info = persistence._compress_data(state)
        
        assert info["method"] == method.value
        assert info["compressed_size"] < info["original_size"]
        assert info["dtype"] == str(state.dtype)
        assert info["shape"] == state.shape

def test_holographic_corrections(persistence):
    """Test holographic corrections during save/load."""
    state = np.random.random(100) + 1j * np.random.random(100)
    state /= np.sqrt(np.sum(np.abs(state)**2))
    time = 1.0
    
    # Save with corrections
    state_path = persistence.save_state(state, {"time": time}, time)
    
    # Load and verify corrections
    loaded_state, _ = persistence.load_state(
        version_id=state_path.stem.replace("state_", "")
    )
    
    expected_norm = np.exp(-INFORMATION_GENERATION_RATE * time)
    actual_norm = np.sqrt(np.sum(np.abs(loaded_state)**2))
    
    assert np.abs(actual_norm - expected_norm) < 1e-10 