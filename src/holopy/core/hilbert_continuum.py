"""
HilbertContinuum implementation for dual continuum quantum states.
"""
from ast import List
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply

from holopy.core.propagator import DualContinuumPropagator
from holopy.inference.active_inference import ActiveInferenceEngine, PredictionMetrics
from holopy.metrics.collectors import MetricsCollector
from holopy.metrics.validation_suite import HolographicValidationSuite
from .hilbert import HilbertSpace
from ..config.constants import BOLTZMANN_CONSTANT, COUPLING_CONSTANT, INFORMATION_GENERATION_RATE, CRITICAL_THRESHOLD, SPEED_OF_LIGHT, PLANCK_CONSTANT, E8_DIMENSION, TOTAL_DIMENSION
from .information_hierarchy import InformationHierarchyProcessor
from holopy.visualization.state_visualizer import HolographicVisualizer
from holopy.optimization.performance_optimizer import PerformanceOptimizer
import logging

logger = logging.getLogger(__name__)

@dataclass
class DualState:
    """Container for quantum and classical states."""
    quantum_state: np.ndarray
    classical_density: np.ndarray
    time: float
    coupling_strength: float
    coherence_hierarchy: List[float]
    information_content: float

class HilbertContinuum:
    """Enhanced HilbertContinuum with information hierarchy processing."""
    
    def __init__(
        self,
        hilbert_space: HilbertSpace,
        spatial_points: int,
        spatial_extent: float,
        enable_active_inference: bool = True,
        enable_hierarchy: bool = True,
        metrics_dir: Optional[Path] = None,
        dt: float = 0.01
    ):
        self.hilbert = hilbert_space
        self.spatial_points = spatial_points
        self.spatial_extent = spatial_extent
        
        # Initialize matter and antimatter states
        self.matter_wavefunction = None
        self.antimatter_wavefunction = None
        
        # State tracking DataFrame
        self.metrics_df = pd.DataFrame(columns=[
            'time',
            'density',
            'temperature', 
            'entropy',
            'information_content',
            'processing_rate',
            'coherence',
            'entanglement',
            'phase',
            'energy',
            'coupling_strength',
            'information_flow',
            'stability_measure'
        ])
        
        # Initialize propagator and inference engine
        self.propagator = DualContinuumPropagator(
            spatial_points=spatial_points,
            spatial_extent=spatial_extent
        )
        
        self.inference_engine = (
            ActiveInferenceEngine(
                spatial_points=spatial_points,
                dt=0.01,  # Default timestep
                prediction_horizon=10
            ) if enable_active_inference else None
        )
        
        # Initialize information hierarchy
        self.hierarchy_processor = (
            InformationHierarchyProcessor(
                spatial_points=spatial_points,
                spatial_extent=spatial_extent
            ) if enable_hierarchy else None
        )
        
        # Initialize validation suite
        self.validator = HolographicValidationSuite()
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector(
            output_dir=metrics_dir
        )
        
        # Initialize optimization components
        self.optimizer = PerformanceOptimizer(spatial_points)
        self.visualizer = HolographicVisualizer(spatial_points, spatial_extent)
        
        # Cache frequently used values
        self.dt = dt
        self.cached_propagator = self.optimizer.get_cached_propagator(dt, spatial_extent)
        
        logger.info(
            f"Initialized enhanced HilbertContinuum with {spatial_points} points"
        )
    
    def create_initial_state(self) -> None:
        """Initialize dual continuum quantum states."""
        try:
            # Create spatial grid
            x = np.linspace(-self.spatial_extent/2, self.spatial_extent/2, self.spatial_points)
            sigma = self.spatial_extent/20
            
            # Initial Gaussian state
            psi = np.exp(-x**2/(2*sigma**2))
            psi /= np.sqrt(np.sum(np.abs(psi)**2))
            
            # Project onto holographic basis
            psi = self.hilbert.project_state(psi)
            
            # Initialize matter and antimatter states
            self.matter_wavefunction = psi.copy()
            self.antimatter_wavefunction = psi.copy()
            
            # Record initial metrics
            self._update_metrics(0.0)
            
            logger.info("Created initial dual continuum state")
            
        except Exception as e:
            logger.error(f"Failed to create initial state: {str(e)}")
            raise
    
    def evolve(self, dt: float) -> None:
        """Enhanced evolution with optimizations."""
        try:
            # Use optimized evolution
            matter_new = self.optimizer.optimized_evolution_step(
                self.matter_wavefunction,
                self.cached_propagator,
                dt
            )
            
            antimatter_new = self.optimizer.optimized_evolution_step(
                self.antimatter_wavefunction,
                self.cached_propagator,
                dt
            )
            
            # Calculate coupling with optimization
            coupling = self.optimizer.optimized_coupling_calculation(
                matter_new,
                antimatter_new
            )
            
            # Update states
            self.matter_wavefunction = matter_new
            self.antimatter_wavefunction = antimatter_new
            
            # Update metrics
            metrics = self._calculate_metrics(dt)
            self.metrics_df = self.metrics_df.append(metrics, ignore_index=True)
            
            # Generate visualizations if needed
            if self.visualization_enabled:
                self.visualizer.plot_dual_continuum_state(
                    self.matter_wavefunction,
                    self.antimatter_wavefunction,
                    self.metrics_df.iloc[-1].time,
                    save_path=self.plot_dir / f"state_{len(self.metrics_df)}.png"
                )
            
        except Exception as e:
            logger.error(f"Evolution step failed: {str(e)}")
            raise
    
    def _update_metrics(self, time: float, pred_metrics: Optional[PredictionMetrics] = None) -> None:
        """Update state metrics with comprehensive tracking."""
        try:
            # Calculate basic observables
            matter_density = np.abs(self.matter_wavefunction)**2
            antimatter_density = np.abs(self.antimatter_wavefunction)**2
            
            # Calculate information measures
            entropy = self.hilbert.calculate_entropy(self.matter_wavefunction)
            information = -np.sum(matter_density * np.log2(matter_density + 1e-10))
            
            # Calculate energy and coupling terms
            energy = self.hilbert.calculate_energy(self.matter_wavefunction)
            coupling = self._calculate_coupling_strength()
            
            metrics = {
                'time': time,
                'density': np.sum(matter_density),
                'entropy': entropy,
                'information_content': information,
                'processing_rate': INFORMATION_GENERATION_RATE,
                'coherence': self.get_coherence(),
                'energy': energy,
                'coupling_strength': coupling,
                'stability_measure': np.abs(np.vdot(
                    self.matter_wavefunction,
                    self.matter_wavefunction
                )),
                'entanglement': self._calculate_entanglement(),
                'phase': np.angle(np.sum(self.matter_wavefunction)),
                'information_flow': INFORMATION_GENERATION_RATE * entropy,
                'temperature': self._calculate_temperature()
            }
            
            # Add prediction metrics if available
            if pred_metrics is not None:
                metrics.update({
                    'prediction_error': pred_metrics.prediction_error,
                    'prediction_confidence': pred_metrics.confidence,
                    'information_gain': pred_metrics.information_gain
                })
            
            self.metrics_df = pd.concat([
                self.metrics_df,
                pd.DataFrame([metrics])
            ], ignore_index=True)
            
            logger.debug(f"Updated metrics at t={time:.6f}")
            
        except Exception as e:
            logger.error(f"Metrics update failed: {str(e)}")
            raise
    
    def get_coherence(self) -> float:
        """Calculate cross-continuum coherence with holographic corrections."""
        try:
            # Calculate basic overlap
            overlap = np.abs(np.sum(
                np.conj(self.matter_wavefunction) * 
                self.antimatter_wavefunction
            ))
            
            # Apply holographic corrections
            coherence = overlap / self.spatial_points
            
            # Include active inference effects
            coherence *= (1 + self._calculate_active_inference_term())
            
            return coherence
            
        except Exception as e:
            logger.error(f"Coherence calculation failed: {str(e)}")
            raise
    
    def _calculate_active_inference_term(self) -> float:
        """Calculate active inference contribution to coherence."""
        try:
            # Calculate effective phase difference
            phase_diff = np.angle(
                self.matter_wavefunction * 
                np.conj(self.antimatter_wavefunction)
            )
            
            # Calculate information gradient
            info_gradient = np.gradient(
                -np.abs(self.matter_wavefunction)**2 * 
                np.log2(np.abs(self.matter_wavefunction)**2 + 1e-10)
            )
            
            # Combine phase and information terms
            active_term = np.sum(np.abs(phase_diff * info_gradient))
            return active_term / (4 * np.pi)
            
        except Exception as e:
            logger.error(f"Active inference calculation failed: {str(e)}")
            raise
            
    def _calculate_temperature(self) -> float:
        """Calculate effective temperature from energy distribution."""
        try:
            energies = np.abs(np.fft.fft(self.matter_wavefunction))**2
            mean_energy = np.sum(energies * np.arange(len(energies))) / np.sum(energies)
            return mean_energy / BOLTZMANN_CONSTANT
            
        except Exception as e:
            logger.error(f"Temperature calculation failed: {str(e)}")
            raise
    
    def _calculate_coupling_strength(self) -> float:
        """Calculate matter-antimatter coupling strength."""
        try:
            rho_c = np.outer(
                self.matter_wavefunction,
                np.conj(self.antimatter_wavefunction)
            )
            return np.abs(np.trace(rho_c)) / self.spatial_points
            
        except Exception as e:
            logger.error(f"Coupling strength calculation failed: {str(e)}")
            raise
    
    def generate_analysis_report(self) -> Dict[str, plt.Figure]:
        """Generate comprehensive analysis report."""
        return self.metrics_collector.generate_report()
    
    def _evolve_quantum_state(
        self,
        quantum_state: np.ndarray,
        classical_density: np.ndarray,
        propagator: np.ndarray
    ) -> np.ndarray:
        """Evolve quantum state with coupling."""
        try:
            # Apply propagator evolution
            evolved_state = self.propagator.evolve_quantum_state(
                quantum_state,
                classical_density,
                self.state_history[-1].time
            )
            
            # Apply holographic constraints
            evolved_state = self._apply_holographic_constraints(
                evolved_state,
                classical_density
            )
            
            return evolved_state
            
        except Exception as e:
            logger.error(f"Quantum evolution failed: {str(e)}")
            raise
    
    def _evolve_classical_density(
        self,
        classical_density: np.ndarray,
        quantum_state: np.ndarray,
        propagator: np.ndarray
    ) -> np.ndarray:
        """Evolve classical density with quantum backreaction."""
        try:
            # Apply propagator evolution
            evolved_density = self.propagator.evolve_classical_density(
                classical_density,
                quantum_state,
                self.state_history[-1].time
            )
            
            # Apply holographic bounds
            evolved_density = self._apply_holographic_bounds(
                evolved_density,
                quantum_state
            )
            
            return evolved_density
            
        except Exception as e:
            logger.error(f"Classical evolution failed: {str(e)}")
            raise
    
    def _apply_holographic_constraints(
        self,
        quantum_state: np.ndarray,
        classical_density: np.ndarray
    ) -> np.ndarray:
        """Apply holographic constraints to quantum state."""
        try:
            # Calculate information bound
            max_information = self._calculate_holographic_bound()
            
            # Calculate current information
            current_information = self._calculate_information_content(
                quantum_state,
                classical_density
            )
            
            if current_information > max_information:
                # Scale state to satisfy bound
                scale_factor = np.sqrt(max_information / current_information)
                quantum_state *= scale_factor
            
            return quantum_state
            
        except Exception as e:
            logger.error(f"Constraint application failed: {str(e)}")
            raise
    
    def _apply_holographic_bounds(
        self,
        classical_density: np.ndarray,
        quantum_state: np.ndarray
    ) -> np.ndarray:
        """Apply holographic bounds to classical density."""
        try:
            # Calculate maximum allowed density
            max_density = 1.0 / (
                PLANCK_CONSTANT * np.sqrt(INFORMATION_GENERATION_RATE)
            )
            
            # Apply bound
            classical_density = np.minimum(classical_density, max_density)
            
            # Renormalize
            classical_density /= np.sum(classical_density) * self.dx
            
            return classical_density
            
        except Exception as e:
            logger.error(f"Bound application failed: {str(e)}")
            raise