import profile
import pytest
import numpy as np
import time
import psutil
import cProfile
import pstats
from src.holopy.core.hilbert_continuum import HilbertContinuum
from src.holopy.core.propagator import DualContinuumPropagator
from src.holopy.config.constants import (
    INFORMATION_GENERATION_RATE,
    MAX_MEMORY_USAGE,
    TARGET_EVOLUTION_TIME
)
import logging

logger = logging.getLogger(__name__)

class TestSystemPerformance:
    """Performance testing suite for holographic system."""
    
    @pytest.fixture(autouse=True)
    def setup_profiling(self):
        """Setup performance testing environment."""
        self.profiler = cProfile.Profile()
        self.spatial_points_list = [128, 256, 512, 1024]
        self.evolution_steps = 1000
        self.results = {}
        
    def test_evolution_scaling(self):
        """Test computational scaling with system size."""
        scaling_results = {}
        
        for n_points in self.spatial_points_list:
            # Initialize system
            continuum = HilbertContinuum(
                spatial_points=n_points,
                spatial_extent=10.0,
                dt=0.001
            )
            
            # Measure evolution time
            start_time = time.perf_counter()
            self.profiler.enable()
            
            state = continuum.initialize_state()
            evolved_state = continuum.evolve_state(
                state,
                steps=self.evolution_steps
            )
            
            self.profiler.disable()
            end_time = time.perf_counter()
            
            # Calculate metrics
            evolution_time = end_time - start_time
            stats = pstats.Stats(self.profiler)
            
            scaling_results[n_points] = {
                'evolution_time': evolution_time,
                'steps_per_second': self.evolution_steps / evolution_time,
                'memory_per_point': psutil.Process().memory_info().rss / n_points
            }
            
            # Validate performance
            assert evolution_time / self.evolution_steps < TARGET_EVOLUTION_TIME
            
        self.results['scaling'] = scaling_results
        self._analyze_scaling_results()
    
    @profile
    def test_memory_usage(self):
        """Test memory usage patterns."""
        memory_results = {}
        
        for n_points in self.spatial_points_list:
            # Initialize system
            continuum = HilbertContinuum(
                spatial_points=n_points,
                spatial_extent=10.0,
                dt=0.001
            )
            
            # Measure baseline memory
            baseline_memory = psutil.Process().memory_info().rss
            
            # Evolution loop
            state = continuum.initialize_state()
            for _ in range(10):  # Shorter test for memory profiling
                state = continuum.evolve_state(state)
            
            # Measure peak memory
            peak_memory = psutil.Process().memory_info().rss
            
            memory_results[n_points] = {
                'baseline_memory': baseline_memory,
                'peak_memory': peak_memory,
                'memory_increase': peak_memory - baseline_memory,
                'memory_per_state': (peak_memory - baseline_memory) / 10
            }
            
            # Validate memory usage
            assert peak_memory < MAX_MEMORY_USAGE
            
        self.results['memory'] = memory_results
        self._analyze_memory_results()
    
    def test_propagator_performance(self):
        """Test propagator computational efficiency."""
        propagator_results = {}
        
        for n_points in self.spatial_points_list:
            propagator = DualContinuumPropagator(
                spatial_points=n_points,
                spatial_extent=10.0,
                dt=0.001
            )
            
            # Generate test states
            quantum_state = np.random.normal(
                0, 1, n_points
            ) + 1j * np.random.normal(0, 1, n_points)
            quantum_state /= np.linalg.norm(quantum_state)
            
            classical_density = np.abs(
                np.random.normal(0, 1, n_points)
            )
            classical_density /= np.sum(classical_density)
            
            # Measure propagator performance
            start_time = time.perf_counter()
            
            for _ in range(100):  # Propagator-specific test
                quantum_state = propagator.evolve_quantum_state(
                    quantum_state,
                    classical_density,
                    0.0
                )
                classical_density = propagator.evolve_classical_density(
                    classical_density,
                    quantum_state,
                    0.0
                )
            
            end_time = time.perf_counter()
            
            propagator_results[n_points] = {
                'time_per_step': (end_time - start_time) / 100,
                'steps_per_second': 100 / (end_time - start_time)
            }
            
        self.results['propagator'] = propagator_results
        self._analyze_propagator_results()
    
    def _analyze_scaling_results(self):
        """Analyze computational scaling behavior."""
        points = np.array(list(self.results['scaling'].keys()))
        times = np.array([
            r['evolution_time']
            for r in self.results['scaling'].values()
        ])
        
        # Fit scaling law
        log_points = np.log(points)
        log_times = np.log(times)
        scaling_exponent = np.polyfit(log_points, log_times, 1)[0]
        
        logger.info(f"System scaling exponent: {scaling_exponent:.3f}")
        assert scaling_exponent < 2.0  # Should be better than O(N²)
    
    def _analyze_memory_results(self):
        """Analyze memory usage patterns."""
        for n_points, results in self.results['memory'].items():
            memory_per_point = results['memory_increase'] / n_points
            logger.info(
                f"Memory per point ({n_points} points): "
                f"{memory_per_point/1024:.2f} KB"
            )
            
            # Check memory scaling
            assert memory_per_point < 1024 * 10  # Less than 10KB per point
    
    def _analyze_propagator_results(self):
        """Analyze propagator performance."""
        for n_points, results in self.results['propagator'].items():
            steps_per_second = results['steps_per_second']
            logger.info(
                f"Propagator performance ({n_points} points): "
                f"{steps_per_second:.1f} steps/second"
            )
            
            # Check minimum performance
            assert steps_per_second > 100  # Minimum 100 steps per second 