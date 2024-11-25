import pytest
from pathlib import Path
from holopy.persistence.state_manager import StatePersistence

def test_state_persistence_initialization(tmp_path):
    persistence = StatePersistence(base_path=tmp_path)
    assert persistence.base_path == tmp_path
    assert persistence.states_path.exists()
    assert persistence.metrics_path.exists()
    assert persistence.version_file.exists()

def test_storage_initialization(tmp_path):
    persistence = StatePersistence(base_path=tmp_path)
    
    # Check directory structure
    assert (tmp_path / "states").exists()
    assert (tmp_path / "metrics").exists()
    assert (tmp_path / "versions.json").exists() 