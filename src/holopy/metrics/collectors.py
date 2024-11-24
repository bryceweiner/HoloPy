"""
Metrics collection module for holographic simulation measurements.
"""
from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from ..config.constants import (
    INFORMATION_GENERATION_RATE,
    PLANCK_CONSTANT,
    CRITICAL_THRESHOLD
)
import logging
from ..core.classical_states import ContinuumState

logger = logging.getLogger(__name__)

@dataclass
class StateMetrics:
    """Container for quantum state metrics."""
    time: float
    coherence: float
    entropy: float
    information_content: float
    integration_measure: float
    energy: float

class MetricsCollector:
    """Collects and manages metrics from holographic simulation."""
    
    def __init__(self, cache_size: int = 1000):
        """Initialize metrics collector."""
        self.cache_size = cache_size
        self.metrics_history: List[StateMetrics] = []
        self.performance_metrics: Dict[str, List[float]] = {
            'computation_time': [],
            'memory_usage': [],
            'cache_hits': [],
            'cache_misses': []
        }
        
        logger.info(f"Initialized MetricsCollector with cache_size={cache_size}")
    
    def collect_state_metrics(
        self,
        state: np.ndarray,
        time: float,
        density_matrix: Optional[np.ndarray] = None
    ) -> StateMetrics:
        """
        Collect metrics from current quantum state.
        
        Args:
            state: Current quantum state vector
            time: Current simulation time
            density_matrix: Optional density matrix for mixed states
        """
        # Calculate coherence
        coherence = self._calculate_coherence(state, time)
        
        # Calculate von Neumann entropy with holographic corrections
        entropy = self._calculate_entropy(
            density_matrix if density_matrix is not None 
            else np.outer(state, state.conj())
        )
        
        # Calculate information content
        info_content = self._calculate_information_content(state)
        
        # Calculate integration measure (Φ)
        integration = self._calculate_integration_measure(state)
        
        # Calculate energy
        energy = self._calculate_energy(state)
        
        metrics = StateMetrics(
            time=time,
            coherence=coherence,
            entropy=entropy,
            information_content=info_content,
            integration_measure=integration,
            energy=energy
        )
        
        self._store_metrics(metrics)
        return metrics
    
    def _calculate_coherence(self, state: np.ndarray, time: float) -> float:
        """Calculate quantum coherence with holographic corrections."""
        # Implementation based on equation from math.tex:2730-2731
        return float(np.abs(np.vdot(state, state)) * np.exp(-INFORMATION_GENERATION_RATE * time))
    
    def _calculate_entropy(self, density_matrix: np.ndarray) -> float:
        """Calculate von Neumann entropy with holographic corrections."""
        # Implementation based on equation from math.tex:4907-4909
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        entropy = 0.0
        for eig in eigenvalues:
            if eig > 1e-10:  # Numerical threshold
                entropy -= eig * np.log2(eig)
        return entropy * np.exp(-INFORMATION_GENERATION_RATE * time)
    
    def _calculate_information_content(self, state: np.ndarray) -> float:
        """Calculate total information content."""
        # Implementation based on equation from math.tex:4914-4915
        return -np.sum(np.abs(state)**2 * np.log2(np.abs(state)**2 + 1e-10))
    
    def _calculate_integration_measure(self, state: np.ndarray) -> float:
        """Calculate integration measure Φ."""
        # Implementation based on equation from knowledgebase/knowledgebase.md:133
        rho = np.outer(state, state.conj())
        size = len(state)
        phi = 0.0
        
        for i in range(size):
            for j in range(size):
                if rho[i,j] > 1e-10:
                    phi -= rho[i,j] * np.log2(rho[i,j])
                    
        return phi
    
    def _calculate_energy(self, state: np.ndarray) -> float:
        """Calculate energy expectation value."""
        # Implementation based on equation from math.tex:2721-2723
        k = np.fft.fftfreq(len(state))
        psi_k = np.fft.fft(state)
        energy = np.sum(
            (k**2 + 1j * INFORMATION_GENERATION_RATE * k) 
            * np.abs(psi_k)**2
        )
        return float(np.real(energy))
    
    def _store_metrics(self, metrics: StateMetrics) -> None:
        """Store metrics in history with cache management."""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.cache_size:
            self.metrics_history.pop(0)
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Convert metrics history to pandas DataFrame."""
        return pd.DataFrame([vars(m) for m in self.metrics_history])
    
    def record_performance_metric(
        self,
        metric_type: str,
        value: float
    ) -> None:
        """Record a performance metric."""
        if metric_type in self.performance_metrics:
            self.performance_metrics[metric_type].append(value)
            if len(self.performance_metrics[metric_type]) > self.cache_size:
                self.performance_metrics[metric_type].pop(0) 
    
    def collect_classical_metrics(
        self,
        classical_state: ContinuumState
    ) -> None:
        """Collect metrics for classical observables."""
        metrics = {
            'temperature': classical_state.temperature,
            'classical_entropy': classical_state.entropy,
            'density_mean': np.mean(classical_state.density),
            'density_std': np.std(classical_state.density)
        }
        
        self.classical_metrics_history.append(metrics)
        
        # Log metrics
        logger.info(
            f"Classical metrics at t={classical_state.time:.2e}: "
            f"T={metrics['temperature']:.2e}, "
            f"S={metrics['classical_entropy']:.2e}"
        )