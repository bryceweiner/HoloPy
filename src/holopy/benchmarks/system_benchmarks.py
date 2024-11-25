from typing import Dict, List, Optional
import numpy as np
import time
import json
from dataclasses import dataclass
from pathlib import Path
from ..core.hilbert_continuum import DualState, HilbertContinuum, HilbertSpace
from ..config.constants import (
    BENCHMARK_CONFIGS,
    INFORMATION_GENERATION_RATE,
    PERFORMANCE_TARGETS
)
import logging

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    config_name: str
    spatial_points: int
    evolution_steps: int
    total_time: float
    steps_per_second: float
    memory_usage: float
    accuracy_metrics: Dict[str, float]

class SystemBenchmark:
    """Benchmarking system for holographic simulations."""
    
    def __init__(
        self,
        results_dir: Optional[Path] = None
    ):
        self.results_dir = results_dir or Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.results: List[BenchmarkResult] = []
        
    def run_benchmarks(
        self,
        configs: Optional[List[Dict]] = None
    ) -> List[BenchmarkResult]:
        """Run complete benchmark suite."""
        try:
            configs = configs or BENCHMARK_CONFIGS
            
            for config in configs:
                logger.info(f"Running benchmark: {config['name']}")
                
                result = self._run_single_benchmark(config)
                self.results.append(result)
                
                # Save intermediate results
                self._save_results()
                
                # Validate against targets
                self._validate_performance(result)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Benchmark suite failed: {str(e)}")
            raise
    
    def _run_single_benchmark(
        self,
        config: Dict
    ) -> BenchmarkResult:
        """Run single benchmark configuration."""
        try:
            # Initialize Hilbert space first
            hilbert_space = HilbertSpace(
                spatial_points=config['spatial_points'],
                spatial_extent=config['spatial_extent']
            )
            
            # Initialize continuum with proper arguments
            continuum = HilbertContinuum(
                spatial_points=config['spatial_points'],
                spatial_extent=config['spatial_extent'],
                hilbert_space=hilbert_space,
                dt=config['dt']
            )
            
            # Initialize state
            state = continuum.initialize_state()
            
            # Evolution loop
            start_time = time.perf_counter()
            start_memory = self._get_memory_usage()
            
            for _ in range(config['evolution_steps']):
                state = continuum.evolve_state(state)
            
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            # Calculate metrics
            total_time = end_time - start_time
            steps_per_second = config['evolution_steps'] / total_time
            memory_usage = end_memory - start_memory
            
            # Calculate accuracy metrics
            accuracy_metrics = self._calculate_accuracy_metrics(
                continuum,
                state
            )
            
            return BenchmarkResult(
                config_name=config['name'],
                spatial_points=config['spatial_points'],
                evolution_steps=config['evolution_steps'],
                total_time=total_time,
                steps_per_second=steps_per_second,
                memory_usage=memory_usage,
                accuracy_metrics=accuracy_metrics
            )
            
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            raise
    
    def _calculate_accuracy_metrics(
        self,
        continuum: HilbertContinuum,
        final_state: 'DualState'
    ) -> Dict[str, float]:
        """Calculate accuracy and conservation metrics."""
        try:
            # Calculate information conservation
            initial_info = continuum.state_history[0].information_content
            expected_info = initial_info * np.exp(
                -INFORMATION_GENERATION_RATE * final_state.time
            )
            info_error = abs(
                final_state.information_content - expected_info
            ) / expected_info
            
            # Calculate normalization error
            quantum_norm = np.sum(np.abs(final_state.quantum_state)**2)
            classical_norm = np.sum(
                final_state.classical_density
            ) * continuum.dx
            
            norm_error = max(
                abs(quantum_norm - 1.0),
                abs(classical_norm - 1.0)
            )
            
            return {
                'information_conservation_error': float(info_error),
                'normalization_error': float(norm_error),
                'final_coupling_strength': float(
                    final_state.coupling_strength
                )
            }
            
        except Exception as e:
            logger.error(f"Accuracy calculation failed: {str(e)}")
            raise
    
    def _save_results(self):
        """Save benchmark results to file."""
        try:
            results_file = self.results_dir / "benchmark_results.json"
            
            results_data = [
                {
                    'config_name': r.config_name,
                    'spatial_points': r.spatial_points,
                    'evolution_steps': r.evolution_steps,
                    'total_time': r.total_time,
                    'steps_per_second': r.steps_per_second,
                    'memory_usage': r.memory_usage,
                    'accuracy_metrics': r.accuracy_metrics
                }
                for r in self.results
            ]
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Results saving failed: {str(e)}")
            raise
    
    def _validate_performance(
        self,
        result: BenchmarkResult
    ):
        """Validate performance against targets."""
        try:
            target = PERFORMANCE_TARGETS[result.config_name]
            
            assert result.steps_per_second >= target['min_steps_per_second']
            assert result.memory_usage <= target['max_memory_usage']
            assert result.accuracy_metrics['information_conservation_error'] <= target['max_info_error']
            assert result.accuracy_metrics['normalization_error'] <= target['max_norm_error']
            
            logger.info(
                f"Benchmark {result.config_name} passed all performance targets"
            )
            
        except Exception as e:
            logger.error(f"Performance validation failed: {str(e)}")
            raise