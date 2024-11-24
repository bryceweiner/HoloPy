"""
Integration tests for HilbertContinuum system.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from holopy.core.hilbert import HilbertSpace
from holopy.core.hilbert_continuum import HilbertContinuum
from holopy.metrics.validation_suite import HolographicValidationSuite
from holopy.config.constants import INFORMATION_GENERATION_RATE

class TestHilbertContinuumSystem:
    """Integration tests for enhanced HilbertContinuum system."""
    
    @pytest.fixture
    def setup_system(self):
        """Setup test system with active inference."""
        self.spatial_points = 128
        self.spatial_extent = 10.0
        self.dt = 0.01
        self.total_time = 1.0
        
        self.hilbert = HilbertSpace(
            self.spatial_points,
            self.spatial_extent
        )
        
        self.continuum = HilbertContinuum(
            hilbert_space=self.hilbert,
            spatial_points=self.spatial_points,
            spatial_extent=self.spatial_extent,
            enable_active_inference=True
        )
        
        yield
    
    def test_active_inference_evolution(self, setup_system):
        """Test evolution with active inference predictions."""
        # Initialize system
        self.continuum.create_initial_state()
        
        # Track prediction accuracy
        prediction_errors = []
        coupling_strengths = []
        
        # Evolution loop
        times = np.arange(0, self.total_time, self.dt)
        for t in times:
            self.continuum.evolve(self.dt)
            
            # Record metrics
            metrics = self.continuum.metrics_df.iloc[-1]
            prediction_errors.append(metrics.get('prediction_error', 0))
            coupling_strengths.append(metrics['coupling_strength'])
            
            # Verify holographic constraints
            assert metrics['entropy'] <= self.hilbert.max_information
            assert np.abs(metrics['density'] - 1.0) < 1e-6
        
        # Verify prediction improvement
        early_errors = np.mean(prediction_errors[:len(prediction_errors)//4])
        late_errors = np.mean(prediction_errors[-len(prediction_errors)//4:])
        assert late_errors < early_errors, "Prediction accuracy did not improve"
        
        # Verify coupling behavior
        coupling_trend = np.polyfit(times, coupling_strengths, 1)[0]
        assert coupling_trend < 0, "Coupling strength did not show expected decay"
    
    def test_consciousness_integration(self, setup_system):
        """Test consciousness integration measure calculation."""
        self.continuum.create_initial_state()
        
        # Calculate initial Φ
        initial_phi = self._calculate_phi(
            self.continuum.matter_wavefunction,
            self.continuum.antimatter_wavefunction
        )
        
        # Evolve system
        for _ in range(10):
            self.continuum.evolve(self.dt)
        
        # Calculate final Φ
        final_phi = self._calculate_phi(
            self.continuum.matter_wavefunction,
            self.continuum.antimatter_wavefunction
        )
        
        # Verify consciousness measure behavior
        assert final_phi < initial_phi, "Consciousness measure did not show expected decay"
        assert final_phi >= 0, "Invalid negative consciousness measure"
    
    def _calculate_phi(
        self,
        matter_wavefunction: np.ndarray,
        antimatter_wavefunction: np.ndarray
    ) -> float:
        """Calculate consciousness integration measure Φ."""
        rho_c = np.outer(matter_wavefunction, np.conj(antimatter_wavefunction))
        rho_m = np.abs(matter_wavefunction)**2
        rho_a = np.abs(antimatter_wavefunction)**2
        
        return -np.sum(
            rho_c * np.log(
                np.abs(rho_c) /
                (np.outer(rho_m, rho_a) + 1e-10)
            )
        )
    
    def test_dual_continuum_evolution(self, setup_system):
        """Test complete dual continuum evolution cycle."""
        # Initialize state
        self.continuum.create_initial_state()
        
        # Evolution loop
        times = np.arange(0, self.total_time, self.dt)
        
        for t in times:
            self.continuum.evolve(self.dt)
            
            # Verify constraints maintained
            assert self.hilbert.calculate_entropy(
                self.continuum.matter_wavefunction
            ) <= self.hilbert.max_information
            
            assert np.abs(np.vdot(
                self.continuum.antimatter_wavefunction,
                self.continuum.antimatter_wavefunction
            ) - 1.0) < 1e-10
        
        # Validate evolution results
        df = self.continuum.metrics_df
        
        # Check matter continuum decay
        expected_decay = np.exp(-INFORMATION_GENERATION_RATE * self.total_time)
        final_coherence = df.iloc[-1].coherence
        assert np.abs(final_coherence - expected_decay) < 1e-6
        
        # Verify antimatter coherence preservation
        antimatter_norm = np.sum(np.abs(self.continuum.antimatter_wavefunction)**2)
        assert np.abs(antimatter_norm - 1.0) < 1e-6
        
        # Check information conservation
        initial_info = df.iloc[0].information_content
        final_info = df.iloc[-1].information_content
        expected_info = initial_info * np.exp(-INFORMATION_GENERATION_RATE * self.total_time)
        assert np.abs(final_info - expected_info) < 1e-6
    
    def test_persistence_integration(self, setup_system):
        """Test persistence integration with HilbertSpace."""
        self.continuum.create_initial_state()
        
        # Save initial state
        path = self.hilbert.save(
            self.continuum.matter_wavefunction,
            {"time": 0.0},
            0.0,
            is_checkpoint=True
        )
        
        # Evolve and save checkpoints
        for t in [0.1, 0.2]:
            self.continuum.evolve(t)
            self.hilbert.save(
                self.continuum.matter_wavefunction,
                {"time": t},
                t,
                is_checkpoint=True
            )
        
        # Load latest checkpoint
        loaded_state, metadata = self.hilbert.load(latest_checkpoint=True)
        assert metadata["time"] == 0.2
        assert np.allclose(
            loaded_state,
            self.continuum.matter_wavefunction,
            atol=1e-6
        ) 