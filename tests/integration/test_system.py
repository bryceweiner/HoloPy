"""
Integration tests for holographic simulation system.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from holopy.core.quantum_states import QuantumState
from holopy.core.hilbert import HilbertSpace
from holopy.core.propagator import FieldPropagator
from holopy.metrics.collectors import MetricsCollector
from holopy.utils.persistence import StatePersistence
from holopy.config.constants import INFORMATION_GENERATION_RATE

class TestHolographicSystem:
    @pytest.fixture
    def setup_system(self):
        """Setup complete holographic simulation system."""
        # System parameters
        self.spatial_points = 100
        self.spatial_extent = 1.0
        self.dt = 1e-6
        self.total_time = 1e-3
        
        # Initialize components
        self.hilbert = HilbertSpace(self.spatial_points, self.spatial_extent)
        self.propagator = FieldPropagator(
            self.spatial_points,
            self.dt,
            self.spatial_extent
        )
        self.metrics = MetricsCollector()
        
        # Create temporary directory for persistence
        self.temp_dir = tempfile.mkdtemp()
        self.persistence = StatePersistence(Path(self.temp_dir))
        
        yield
        
        # Cleanup
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_evolution_cycle(self, setup_system):
        """Test complete system evolution cycle with all components."""
        # Create initial state
        state = QuantumState.create_initial_state(
            self.spatial_points,
            self.spatial_extent
        )
        
        # Evolution loop
        times = np.arange(0, self.total_time, self.dt)
        metrics_history = []
        
        for t in times:
            # Evolve state
            state.evolve(self.dt)
            
            # Project onto holographic basis
            state.wavefunction = self.hilbert.project_state(state.wavefunction)
            
            # Apply field propagator
            state.wavefunction = self.propagator.propagate(
                state.wavefunction,
                self.dt
            )
            
            # Collect metrics
            metrics = self.metrics.collect_state_metrics(
                state.wavefunction,
                t
            )
            metrics_history.append(metrics)
            
            # Periodic checkpoint
            if len(metrics_history) % 10 == 0:
                self.persistence.save_state(
                    state.wavefunction,
                    {'time': t},
                    t,
                    is_checkpoint=True
                )
        
        # Validate complete evolution
        df = pd.DataFrame([vars(m) for m in metrics_history])
        
        # Check holographic decay
        expected_coherence = np.exp(-INFORMATION_GENERATION_RATE * self.total_time)
        final_coherence = df.iloc[-1].coherence
        assert np.abs(final_coherence - expected_coherence) < 1e-6
        
        # Check information conservation (from math.tex:4913-4915)
        initial_info = df.iloc[0].information_content
        final_info = df.iloc[-1].information_content
        expected_info = initial_info * np.exp(-INFORMATION_GENERATION_RATE * self.total_time)
        assert np.abs(final_info - expected_info) < 1e-6
        
        # Verify state reconstruction
        reconstructed_state, _ = self.persistence.load_state(latest_checkpoint=True)
        assert np.allclose(reconstructed_state, state.wavefunction, atol=1e-6) 