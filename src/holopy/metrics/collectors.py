"""
Metrics collection system for holographic simulation.
"""
from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
from ..config.constants import (
    INFORMATION_GENERATION_RATE,
    PLANCK_CONSTANT,
    SPEED_OF_LIGHT,
    COUPLING_CONSTANT
)
from .validation_suite import HolographicValidationSuite

logger = logging.getLogger(__name__)

@dataclass
class StateMetrics:
    """Container for comprehensive state metrics."""
    time: float
    density: float
    temperature: float
    entropy: float
    information_content: float
    coherence: float
    energy: float
    processing_rate: float
    stability_measure: float
    phase: float
    entanglement: float
    information_flow: float
    coupling_strength: float

class MetricsCollector:
    """Collects and processes metrics from quantum and classical states."""
    
    def __init__(self, validation_suite: Optional[HolographicValidationSuite] = None):
        self.metrics_history: List[StateMetrics] = []
        self.validation_suite = validation_suite or HolographicValidationSuite()
        self._initialize_tracking()
        logger.info("Initialized MetricsCollector")
    
    def _initialize_tracking(self) -> None:
        """Initialize metrics tracking structures."""
        try:
            self.tracking_df = pd.DataFrame(columns=[
                'time',
                'density',
                'temperature',
                'entropy',
                'information_content',
                'coherence',
                'energy',
                'processing_rate',
                'stability_measure',
                'phase',
                'entanglement',
                'information_flow',
                'coupling_strength'
            ])
            
            logger.debug("Initialized metrics tracking")
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics tracking: {str(e)}")
            raise
    
    def collect_state_metrics(
        self,
        wavefunction: np.ndarray,
        time: float,
        classical_state: Optional[Dict] = None
    ) -> StateMetrics:
        """
        Collect comprehensive metrics from quantum and classical states.
        
        Args:
            wavefunction: Quantum state vector
            time: Current simulation time
            classical_state: Optional classical observables
            
        Returns:
            StateMetrics containing collected metrics
        """
        try:
            # Calculate quantum metrics
            density = np.abs(wavefunction)**2
            phase = np.angle(np.sum(wavefunction))
            energy = self._calculate_energy(wavefunction)
            
            # Calculate information measures
            entropy = self._calculate_entropy(density)
            information = -np.sum(density * np.log2(density + 1e-10))
            
            # Calculate coherence and stability
            coherence = self._calculate_coherence(wavefunction)
            stability = np.abs(np.vdot(wavefunction, wavefunction))
            
            # Include classical observables if provided
            temperature = classical_state.get('temperature', 0.0) if classical_state else 0.0
            coupling = classical_state.get('coupling_strength', 0.0) if classical_state else 0.0
            
            metrics = StateMetrics(
                time=time,
                density=np.sum(density),
                temperature=temperature,
                entropy=entropy,
                information_content=information,
                coherence=coherence,
                energy=energy,
                processing_rate=INFORMATION_GENERATION_RATE,
                stability_measure=stability,
                phase=phase,
                entanglement=self._calculate_entanglement(wavefunction),
                information_flow=INFORMATION_GENERATION_RATE * entropy,
                coupling_strength=coupling
            )
            
            # Update tracking
            self.metrics_history.append(metrics)
            self._update_tracking_df(metrics)
            
            logger.debug(f"Collected metrics at t={time:.6f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {str(e)}")
            raise
    
    def _calculate_energy(self, wavefunction: np.ndarray) -> float:
        """Calculate total energy with holographic corrections."""
        try:
            # Calculate kinetic energy in momentum space
            psi_k = np.fft.fft(wavefunction)
            k = 2 * np.pi * np.fft.fftfreq(len(wavefunction))
            kinetic = np.sum(np.abs(psi_k)**2 * k**2) / 2
            
            # Calculate potential energy with corrections
            density = np.abs(wavefunction)**2
            potential = COUPLING_CONSTANT * np.sum(density * np.arange(len(density))**2)
            
            # Add holographic correction
            correction = INFORMATION_GENERATION_RATE * np.sum(
                density * np.log(density + 1e-10)
            ) / (4 * np.pi)
            
            return kinetic + potential + correction
            
        except Exception as e:
            logger.error(f"Energy calculation failed: {str(e)}")
            raise
    
    def _calculate_entropy(self, density: np.ndarray) -> float:
        """Calculate von Neumann entropy with holographic bound."""
        try:
            entropy = -np.sum(density * np.log(density + 1e-10))
            max_entropy = np.log(len(density))
            
            return min(entropy, max_entropy)
            
        except Exception as e:
            logger.error(f"Entropy calculation failed: {str(e)}")
            raise
    
    def _calculate_coherence(self, wavefunction: np.ndarray) -> float:
        """Calculate quantum coherence measure."""
        try:
            # Calculate off-diagonal elements of density matrix
            density_matrix = np.outer(wavefunction, np.conj(wavefunction))
            coherence = np.sum(np.abs(density_matrix - np.diag(np.diag(density_matrix))))
            
            return coherence
            
        except Exception as e:
            logger.error(f"Coherence calculation failed: {str(e)}")
            raise
    
    def _calculate_entanglement(self, wavefunction: np.ndarray) -> float:
        """Calculate entanglement entropy for bipartite split."""
        try:
            # Reshape for bipartite split
            n = len(wavefunction)
            mid = n // 2
            rho = np.outer(wavefunction, np.conj(wavefunction))
            
            # Calculate reduced density matrix
            rho_a = np.trace(rho.reshape(mid, n//mid, mid, n//mid), axis1=1, axis2=3)
            
            # Calculate entanglement entropy
            eigs = np.linalg.eigvalsh(rho_a)
            eigs = eigs[eigs > 1e-10]
            return -np.sum(eigs * np.log2(eigs))
            
        except Exception as e:
            logger.error(f"Entanglement calculation failed: {str(e)}")
            raise
    
    def _update_tracking_df(self, metrics: StateMetrics) -> None:
        """Update metrics tracking DataFrame."""
        try:
            self.tracking_df = pd.concat([
                self.tracking_df,
                pd.DataFrame([vars(metrics)])
            ], ignore_index=True)
            
        except Exception as e:
            logger.error(f"Failed to update tracking DataFrame: {str(e)}")
            raise