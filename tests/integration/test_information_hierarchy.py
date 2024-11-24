import numpy as np
import pytest

from holopy.core.information_hierarchy import InformationHierarchyProcessor


class TestInformationHierarchy:
    """Test information hierarchy processing system."""
    
    @pytest.fixture
    def setup_hierarchy(self):
        """Setup test hierarchy system."""
        self.spatial_points = 128
        self.spatial_extent = 10.0
        self.dt = 0.01
        self.total_time = 1.0
        
        self.hierarchy = InformationHierarchyProcessor(
            spatial_points=self.spatial_points,
            spatial_extent=self.spatial_extent
        )
        
        # Create test wavefunction
        x = np.linspace(-self.spatial_extent/2, self.spatial_extent/2, self.spatial_points)
        psi = np.exp(-x**2/2)
        self.initial_state = psi / np.sqrt(np.sum(np.abs(psi)**2))
        
        yield
    
    def test_hierarchy_processing(self, setup_hierarchy):
        """Test multi-level information processing."""
        state = self.initial_state.copy()
        processing_rates = []
        coherence_values = []
        
        # Process through hierarchy
        times = np.arange(0, self.total_time, self.dt)
        for t in times:
            state, metrics = self.hierarchy.process_state(state, self.dt)
            
            # Track metrics
            processing_rates.append([
                metrics[f'level_{i}']['processing_rate']
                for i in range(self.hierarchy.num_levels)
            ])
            coherence_values.append([
                metrics[f'level_{i}']['coherence']
                for i in range(self.hierarchy.num_levels)
            ])
        
        # Verify processing rate hierarchy
        rates = np.array(processing_rates)
        for i in range(self.hierarchy.num_levels-1):
            assert np.all(rates[:, i] > rates[:, i+1]), \
                f"Invalid processing rate hierarchy at level {i}"
        
        # Verify coherence decay
        coherence = np.array(coherence_values)
        for i in range(self.hierarchy.num_levels):
            decay_rate = np.polyfit(times, np.log(coherence[:, i]), 1)[0]
            expected_rate = -self.hierarchy.levels[i].processing_rate
            assert np.abs(decay_rate - expected_rate) < 0.1, \
                f"Invalid coherence decay at level {i}" 