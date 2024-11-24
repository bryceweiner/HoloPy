"""Integration tests for visualization system."""
from matplotlib import pyplot as plt
import pytest
import numpy as np
from pathlib import Path
from holopy.visualization.state_visualizer import HolographicVisualizer

class TestVisualizationSystem:
    @pytest.fixture
    def setup_visualizer(self):
        """Setup test visualization system."""
        self.spatial_points = 128
        self.spatial_extent = 10.0
        self.visualizer = HolographicVisualizer(
            self.spatial_points,
            self.spatial_extent
        )
        
        # Create test states
        x = np.linspace(-5, 5, self.spatial_points)
        self.matter = np.exp(-x**2/2)
        self.antimatter = np.exp(-x**2/2) * np.exp(1j * x)
        
        yield
    
    def test_state_visualization(self, setup_visualizer):
        """Test state visualization generation."""
        fig = self.visualizer.plot_dual_continuum_state(
            self.matter,
            self.antimatter,
            0.0
        )
        
        assert fig is not None
        plt.close(fig) 