"""
Performance benchmarks for holographic simulation.
"""
import pytest
import numpy as np
import time
import psutil
import pandas as pd
from holopy.core.quantum_states import QuantumState
from holopy.core.hilbert import HilbertSpace
from holopy.core.propagator import FieldPropagator
from holopy.utils.persistence import StatePersistence
from pathlib import Path

class BenchmarkMetrics:
    def __init__(self):
        self.timings = []
        self.memory_usage = []
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    def record_timing(self, operation: str, duration: float):
        self.timings.append({'operation': operation, 'duration': duration})
    
    def record_memory(self, snapshot: float):
        self.memory_usage.append(snapshot)
    
    def record_cache_event(self, hit: bool):
        if hit:
            self.cache_stats['hits'] += 1
        else:
            self.cache_stats['misses'] += 1

@pytest.mark.benchmark
class TestSystemPerformance:
    @pytest.fixture
    def setup_benchmark(self):
        self.metrics = BenchmarkMetrics()
        return self.metrics
    
    def test_evolution_performance(self, setup_benchmark):
        """Benchmark state evolution performance."""
        spatial_points = 1000  # Larger system for meaningful benchmarks
        dt = 1e-6
        steps = 100
        
        # Initialize system
        start_time = time.perf_counter()
        hilbert = HilbertSpace(spatial_points, 1.0)
        propagator = FieldPropagator(spatial_points, dt, 1.0)
        state = QuantumState.create_initial_state(spatial_points, 1.0)
        setup_time = time.perf_counter() - start_time
        setup_benchmark.record_timing('initialization', setup_time)
        
        # Evolution benchmark
        evolution_times = []
        memory_usage = []
        
        for _ in range(steps):
            # Record memory
            mem = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_usage.append(mem)
            
            # Time evolution step
            start_time = time.perf_counter()
            
            state.evolve(dt)
            state.wavefunction = hilbert.project_state(state.wavefunction)
            state.wavefunction = propagator.propagate(state.wavefunction, dt)
            
            step_time = time.perf_counter() - start_time
            evolution_times.append(step_time)
        
        # Analyze results
        avg_step_time = np.mean(evolution_times)
        max_memory = np.max(memory_usage)
        
        # Performance assertions based on requirements
        assert avg_step_time < 0.1  # Sub-second evolution steps
        assert max_memory < 1000  # Memory usage under 1GB
        
        # Record metrics
        setup_benchmark.record_timing('avg_evolution_step', avg_step_time)
        setup_benchmark.record_memory(max_memory)
    
    def test_persistence_performance(self, setup_benchmark):
        """Benchmark state persistence performance."""
        spatial_points = 1000
        state = QuantumState.create_initial_state(spatial_points, 1.0)
        persistence = StatePersistence(Path('/tmp/benchmark_test'))
        
        # Benchmark save operation
        start_time = time.perf_counter()
        save_path = persistence.save_state(
            state.wavefunction,
            {'benchmark': True},
            0.0
        )
        save_time = time.perf_counter() - start_time
        
        # Benchmark load operation
        start_time = time.perf_counter()
        loaded_state, _ = persistence.load_state(timestamp=0.0)
        load_time = time.perf_counter() - start_time
        
        # Performance assertions
        assert save_time < 1.0  # Sub-second save time
        assert load_time < 1.0  # Sub-second load time
        
        # Record metrics
        setup_benchmark.record_timing('state_save', save_time)
        setup_benchmark.record_timing('state_load', load_time)
        
        #