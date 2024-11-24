import pytest
import numpy as np
from src.holopy.core.hilbert_continuum import HilbertContinuum, DualState
from src.holopy.config.constants import (
    INFORMATION_GENERATION_RATE,
    COUPLING_CONSTANT,
    SPEED_OF_LIGHT,
    PLANCK_CONSTANT
)

@pytest.fixture
def setup_continuum():
    """Setup test continuum system."""
    spatial_points = 128
    spatial_extent = 10.0
    dt = 0.01
    
    continuum = HilbertContinuum(
        spatial_points=spatial_points,
        spatial_extent=spatial_extent,
        dt=dt
    )
    
    return continuum

class TestHilbertContinuum:
    """Test suite for HilbertContinuum class."""
    
    def test_initialization(self, setup_continuum):
        """Test proper initialization of continuum system."""
        continuum = setup_continuum
        
        assert continuum.spatial_points == 128
        assert continuum.spatial_extent == 10.0
        assert continuum.dt == 0.01
        assert len(continuum.metrics_df.columns) == 16
        
    def test_state_initialization(self, setup_continuum):
        """Test initial state creation."""
        continuum = setup_continuum
        
        initial_state = continuum.initialize_state()
        
        # Check state properties
        assert isinstance(initial_state, DualState)
        assert initial_state.time == 0.0
        assert initial_state.coupling_strength == COUPLING_CONSTANT
        assert len(initial_state.coherence_hierarchy) == 3
        
        # Check normalization
        assert np.abs(np.sum(np.abs(initial_state.quantum_state)**2) - 1.0) < 1e-10
        assert np.abs(np.sum(initial_state.classical_density) * continuum.dx - 1.0) < 1e-10
        
    def test_coherence_hierarchy(self, setup_continuum):
        """Test coherence hierarchy calculation."""
        continuum = setup_continuum
        initial_state = continuum.initialize_state()
        
        # Check hierarchy ordering
        C1, C2, C3 = initial_state.coherence_hierarchy
        assert C1 >= C2 >= C3
        assert 0 <= C3 <= C2 <= C1 <= 1
        
    def test_information_conservation(self, setup_continuum):
        """Test information conservation during evolution."""
        continuum = setup_continuum
        initial_state = continuum.initialize_state()
        
        # Evolve system
        evolved_state = continuum.evolve_state(initial_state, steps=10)
        
        # Check information bounds
        initial_info = initial_state.information_content
        final_info = evolved_state.information_content
        expected_info = initial_info * np.exp(-INFORMATION_GENERATION_RATE * evolved_state.time)
        
        assert np.abs(final_info - expected_info) < 1e-6
        
    def test_coupling_dynamics(self, setup_continuum):
        """Test quantum-classical coupling behavior."""
        continuum = setup_continuum
        initial_state = continuum.initialize_state()
        
        # Evolve system
        evolved_state = continuum.evolve_state(initial_state, steps=1)
        
        # Check coupling effects
        quantum_gradient = np.gradient(np.abs(evolved_state.quantum_state)**2)
        classical_gradient = np.gradient(evolved_state.classical_density)
        
        # Verify opposite gradients (attractive coupling)
        correlation = np.corrcoef(quantum_gradient, classical_gradient)[0,1]
        assert correlation < 0
        
    def test_holographic_bounds(self, setup_continuum):
        """Test holographic bound enforcement."""
        continuum = setup_continuum
        initial_state = continuum.initialize_state()
        
        # Evolve system
        evolved_state = continuum.evolve_state(initial_state, steps=5)
        
        # Check maximum information density
        max_density = 1.0 / (PLANCK_CONSTANT * np.sqrt(INFORMATION_GENERATION_RATE))
        assert np.all(evolved_state.classical_density <= max_density * 1.1)  # 10% tolerance 