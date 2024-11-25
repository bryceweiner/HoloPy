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
from typing_extensions import TypeAlias  # For Python < 3.10

from holopy.core.propagator import DualContinuumPropagator
from holopy.inference.active_inference import ActiveInferenceEngine, PredictionMetrics
from holopy.metrics.collectors import MetricsCollector
from holopy.metrics.validation_suite import HolographicValidationSuite
from .hilbert import HilbertSpace
from ..config.constants import BOLTZMANN_CONSTANT, COUPLING_CONSTANT, INFORMATION_GENERATION_RATE, PLANCK_CONSTANT, E8_DIMENSION, TOTAL_DIMENSION
from .information_hierarchy import InformationHierarchyProcessor
from holopy.visualization.state_visualizer import HolographicVisualizer
from holopy.optimization.performance_optimizer import PerformanceOptimizer
import logging

logger = logging.getLogger(__name__)

# Type aliases
FloatArray: TypeAlias = np.ndarray[np.float64]
ComplexArray: TypeAlias = np.ndarray[np.complex128]

@dataclass
class DualState:
    """Container for quantum and classical states."""
    quantum_state: np.ndarray
    classical_density: np.ndarray
    time: float
    coupling_strength: float
    coherence_hierarchy: tuple[float, ...]
    information_content: float

    def __post_init__(self):
        """Validate state initialization."""
        if not isinstance(self.coherence_hierarchy, tuple):
            self.coherence_hierarchy = tuple(self.coherence_hierarchy)

class HilbertContinuum:
    """Enhanced HilbertContinuum with information hierarchy processing."""
    
    def __init__(
        self,
        hilbert_space: HilbertSpace,
        dt: float,
        enable_hierarchy: bool = True,
        output_dir: Optional[Path] = None
    ):
        """Initialize Hilbert continuum.
        
        Args:
            hilbert_space: HilbertSpace instance
            dt: Time step
            enable_hierarchy: Enable hierarchical structure
            output_dir: Output directory for metrics and plots
        """
        self.hilbert_space = hilbert_space
        self.dt = dt
        self.enable_hierarchy = enable_hierarchy
        self.dimension = hilbert_space.dimension
        self.extent = hilbert_space.extent
        self.output_dir = output_dir or Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize matter and antimatter states
        self.matter_wavefunction = None
        self.antimatter_wavefunction = None
        
        # Initialize metrics DataFrame with proper dtypes
        self.metrics_df = pd.DataFrame({
            'time': pd.Series(dtype=float),
            'density': pd.Series(dtype=float),
            'temperature': pd.Series(dtype=float),
            'entropy': pd.Series(dtype=float),
            'information_content': pd.Series(dtype=float),
            'processing_rate': pd.Series(dtype=float),
            'coherence': pd.Series(dtype=float),
            'entanglement': pd.Series(dtype=float),
            'phase': pd.Series(dtype=float),
            'energy': pd.Series(dtype=float),
            'coupling_strength': pd.Series(dtype=float),
            'information_flow': pd.Series(dtype=float),
            'stability_measure': pd.Series(dtype=float)
        })
        
        # Initialize propagator with all required parameters
        self.propagator = DualContinuumPropagator(
            spatial_points=self.dimension,
            spatial_extent=self.extent,
            dt=self.dt
        )
        
        if self.enable_hierarchy:
            self.inference_engine = ActiveInferenceEngine(
                spatial_points=self.dimension,
                dt=self.dt,
                prediction_horizon=10
            )
            
            self.hierarchy_processor = InformationHierarchyProcessor(
                spatial_points=self.dimension,
                spatial_extent=self.extent
            )
        else:
            self.inference_engine = None
            self.hierarchy_processor = None
        
        # Initialize validation suite
        self.validator = HolographicValidationSuite()
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector(
            output_dir=self.output_dir
        )
        
        # Initialize optimization components
        self.optimizer = PerformanceOptimizer(self.dimension)
        self.visualizer = HolographicVisualizer(self.dimension)
        
        # Cache frequently used values
        self.dt = dt
        self.cached_propagator = self.optimizer.get_cached_propagator(dt, self.extent)
        
        logger.info(
            f"Initialized enhanced HilbertContinuum with {self.dimension} points"
        )
    
    def create_initial_state(self) -> None:
        """Initialize dual continuum quantum states."""
        try:
            # Create spatial grid
            x = np.linspace(-self.extent/2, self.extent/2, self.dimension)
            sigma = self.extent/20
            
            # Initial Gaussian state
            psi = np.exp(-x**2/(2*sigma**2))
            psi /= np.sqrt(np.sum(np.abs(psi)**2))
            
            # Project onto holographic basis
            psi = self.hilbert_space.project_state(psi)
            
            # Initialize matter and antimatter states
            self.matter_wavefunction = psi.copy()
            self.antimatter_wavefunction = psi.copy()
            
            # Record initial metrics
            self._update_metrics(0.0)
            
            logger.info("Created initial dual continuum state")
            
        except Exception as e:
            logger.error(f"Failed to create initial state: {str(e)}")
            raise
    
    def evolve(self, *, steps: int) -> None:
        """Evolve the system for a specified number of steps.
        
        Args:
            steps: Number of time steps to evolve
        """
        try:
            if self.matter_wavefunction is None:
                raise ValueError("Initial state not created")
            
            # Initialize metrics collector if not already done
            if not hasattr(self, 'metrics_collector'):
                self.metrics_collector = MetricsCollector(self.output_dir)
            
            # Evolve system
            for _ in range(steps):
                self.evolve_step()
            
            logger.info(f"Completed {steps} evolution steps")
            
        except Exception as e:
            logger.error(f"Evolution failed: {str(e)}")
            raise
    
    def evolve_step(self) -> None:
        """Evolve the quantum state by one time step using split-operator method."""
        try:
            if self.matter_wavefunction is None or self.antimatter_wavefunction is None:
                raise ValueError("Initial state not created")
            
            # Get current time
            current_time = len(self.metrics_df) * self.dt
            
            # 1. Half-step kinetic evolution
            k = self.hilbert_space.k
            exp_k = np.exp(-0.5j * k**2 * self.dt)
            
            psi_k_matter = np.fft.fft(self.matter_wavefunction)
            psi_k_antimatter = np.fft.fft(self.antimatter_wavefunction)
            
            psi_k_matter *= exp_k
            psi_k_antimatter *= exp_k
            
            self.matter_wavefunction = np.fft.ifft(psi_k_matter)
            self.antimatter_wavefunction = np.fft.ifft(psi_k_antimatter)
            
            # 2. Full step potential evolution
            V = self.hilbert_space.potential
            exp_v = np.exp(-1j * V * self.dt)
            
            self.matter_wavefunction *= exp_v
            self.antimatter_wavefunction *= exp_v
            
            # 3. Matter-antimatter coupling
            coupling = self._calculate_coupling_strength()
            interaction = coupling * (
                np.abs(self.antimatter_wavefunction)**2 * self.matter_wavefunction +
                np.abs(self.matter_wavefunction)**2 * self.antimatter_wavefunction
            )
            
            self.matter_wavefunction += -1j * self.dt * interaction
            self.antimatter_wavefunction += -1j * self.dt * np.conj(interaction)
            
            # 4. Half-step kinetic evolution
            psi_k_matter = np.fft.fft(self.matter_wavefunction)
            psi_k_antimatter = np.fft.fft(self.antimatter_wavefunction)
            
            psi_k_matter *= exp_k
            psi_k_antimatter *= exp_k
            
            self.matter_wavefunction = np.fft.ifft(psi_k_matter)
            self.antimatter_wavefunction = np.fft.ifft(psi_k_antimatter)
            
            # 5. Project states to maintain holographic constraints
            self.matter_wavefunction = self.hilbert_space.project_state(
                self.matter_wavefunction
            )
            self.antimatter_wavefunction = self.hilbert_space.project_state(
                self.antimatter_wavefunction
            )
            
            # Update metrics
            self._update_metrics(current_time + self.dt)
            
            logger.debug(f"Evolved state to t={current_time + self.dt:.3f}")
            
        except Exception as e:
            logger.error(f"Evolution step failed: {str(e)}")
            raise
    
    def _update_metrics(self, time: float, pred_metrics: Optional[PredictionMetrics] = None) -> None:
        """Update state metrics with comprehensive tracking."""
        try:
            # Calculate basic observables
            matter_density = np.abs(self.matter_wavefunction)**2
            antimatter_density = np.abs(self.antimatter_wavefunction)**2
            
            # Calculate metrics
            metrics = {
                'time': time,
                'density': np.sum(matter_density),
                'entropy': self.hilbert_space.calculate_entropy(self.matter_wavefunction),
                'information_content': -np.sum(matter_density * np.log2(matter_density + 1e-10)),
                'processing_rate': INFORMATION_GENERATION_RATE,
                'coherence': self.get_coherence(),
                'energy': self.hilbert_space.calculate_energy(self.matter_wavefunction),
                'coupling_strength': self._calculate_coupling_strength(),
                'stability_measure': np.abs(np.vdot(
                    self.matter_wavefunction,
                    self.matter_wavefunction
                )),
                'entanglement': self._calculate_entanglement(),
                'phase': np.angle(np.sum(self.matter_wavefunction)),
                'information_flow': INFORMATION_GENERATION_RATE * self.hilbert_space.calculate_entropy(self.matter_wavefunction),
                'temperature': self._calculate_temperature()
            }
            
            # Update metrics collector
            self.metrics_collector.update(metrics)
            
            # Update DataFrame using loc to avoid concatenation warning
            new_idx = len(self.metrics_df)
            self.metrics_df.loc[new_idx] = metrics
            
        except Exception as e:
            logger.error(f"Metrics update failed: {str(e)}")
            raise
    
    def get_coherence(self) -> float:
        """Calculate quantum coherence measure."""
        try:
            if self.matter_wavefunction is None:
                return 0.0
                
            # Calculate off-diagonal elements of density matrix
            rho = np.outer(
                self.matter_wavefunction,
                np.conj(self.matter_wavefunction)
            )
            
            # l1-norm coherence measure
            coherence = np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))
            
            return float(np.real(coherence))
            
        except Exception as e:
            logger.error(f"Coherence calculation failed: {str(e)}")
            return 0.0
    
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
            return np.abs(np.trace(rho_c)) / self.dimension
            
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
    
    def _calculate_entanglement(self) -> float:
        """Calculate entanglement entropy between matter and antimatter states."""
        try:
            # Calculate reduced density matrix
            rho = np.outer(
                self.matter_wavefunction,
                np.conj(self.antimatter_wavefunction)
            )
            
            # Calculate eigenvalues of reduced density matrix
            eigenvalues = np.linalg.eigvalsh(rho)
            
            # Clean up numerical noise and normalize
            eigenvalues = np.abs(eigenvalues)  # Ensure positive
            eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove negligible values
            if len(eigenvalues) > 0:
                eigenvalues = eigenvalues / np.sum(eigenvalues)  # Normalize
            
            # Calculate von Neumann entropy
            entanglement = -np.sum(
                eigenvalues * np.log2(eigenvalues + 1e-10)
            )
            
            return max(0.0, float(np.real(entanglement)))  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Entanglement calculation failed: {str(e)}")
            return 0.0