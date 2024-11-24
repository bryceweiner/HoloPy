import pytest
import numpy as np
import pandas as pd
from src.holopy.core.hilbert_continuum import HilbertContinuum
from src.holopy.core.propagator import DualContinuumPropagator
from src.holopy.config.constants import (
    INFORMATION_GENERATION_RATE,
    COUPLING_CONSTANT,
    E8_DIMENSION
)

class TestDualContinuumSystem:
    """Integration tests for dual continuum system."""
    
    @pytest.fixture(autouse=True)
    def setup_system(self):
        """Setup test system."""
        self.spatial_points = 256
        self.spatial_extent = 20.0
        self.dt = 0.001
        self.total_time = 0.1
        
        self.continuum = HilbertContinuum(
            spatial_points=self.spatial_points,
            spatial_extent=self.spatial_extent,
            dt=self.dt
        )
        
        self.propagator = DualContinuumPropagator(
            spatial_points=self.spatial_points,
            spatial_extent=self.spatial_extent,
            dt=self.dt
        )
    
    def test_complete_evolution_cycle(self):
        """Test complete evolution cycle with all components."""
        # Initialize system
        initial_state = self.continuum.initialize_state()
        
        # Evolution loop
        current_state = initial_state
        evolution_steps = int(self.total_time / self.dt)
        
        metrics_history = []
        
        for step in range(evolution_steps):
            # Evolve state
            current_state = self.continuum.evolve_state(current_state)
            
            # Collect metrics
            metrics = {
                'time': current_state.time,
                'information_content': current_state.information_content,
                'coupling_strength': current_state.coupling_strength,
                'coherence_primary': current_state.coherence_hierarchy[0],
                'coherence_secondary': current_state.coherence_hierarchy[1],
                'coherence_tertiary': current_state.coherence_hierarchy[2]
            }
            metrics_history.append(metrics)
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics_history)
        
        # Validate evolution
        self._validate_information_decay(df)
        self._validate_coherence_hierarchy(df)
        self._validate_coupling_behavior(df)
        self._validate_conservation_laws(df)
    
    def _validate_information_decay(self, df: pd.DataFrame):
        """Validate holographic information decay."""
        initial_info = df.iloc[0].information_content
        final_info = df.iloc[-1].information_content
        expected_info = initial_info * np.exp(-INFORMATION_GENERATION_RATE * self.total_time)
        
        assert np.abs(final_info - expected_info) < 1e-6
    
    def _validate_coherence_hierarchy(self, df: pd.DataFrame):
        """Validate coherence hierarchy relationships."""
        # Check ordering
        assert np.all(df.coherence_primary >= df.coherence_secondary)
        assert np.all(df.coherence_secondary >= df.coherence_tertiary)
        
        # Check decay rates
        primary_rate = -np.polyfit(df.time, np.log(df.coherence_primary), 1)[0]
        assert np.abs(primary_rate - INFORMATION_GENERATION_RATE/2) < 1e-6
    
    def _validate_coupling_behavior(self, df: pd.DataFrame):
        """Validate coupling strength behavior."""
        # Check coupling bounds
        assert np.all(df.coupling_strength <= COUPLING_CONSTANT)
        assert np.all(df.coupling_strength >= 0)
        
        # Check coupling evolution
        coupling_rate = -np.polyfit(df.time, np.log(df.coupling_strength), 1)[0]
        assert np.abs(coupling_rate - INFORMATION_GENERATION_RATE) < 1e-6
    
    def _validate_conservation_laws(self, df: pd.DataFrame):
        """Validate conservation laws and bounds."""
        # Check E8 dimension bound
        assert np.all(df.information_content <= E8_DIMENSION * np.log(2))
        
        # Check total probability conservation
        final_state = self.continuum.state_history[-1]
        quantum_norm = np.sum(np.abs(final_state.quantum_state)**2)
        classical_norm = np.sum(final_state.classical_density) * self.continuum.dx
        
        assert np.abs(quantum_norm - 1.0) < 1e-10
        assert np.abs(classical_norm - 1.0) < 1e-10 