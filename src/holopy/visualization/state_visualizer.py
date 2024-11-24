"""
Visualization tools for holographic simulation states and metrics.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from pathlib import Path
import seaborn as sns
from ..metrics.collectors import StateMetrics
from ..config.constants import INFORMATION_GENERATION_RATE

class HolographicVisualizer:
    """Visualization tools for holographic states and metrics."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("./visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-darkgrid')
        sns.set_palette("husl")
    
    def plot_state_evolution(
        self,
        states: List[np.ndarray],
        times: List[float],
        x_grid: np.ndarray,
        save: bool = True
    ) -> None:
        """Plot quantum state evolution with holographic corrections."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot wavefunction evolution
        for state, t in zip(states[::10], times[::10]):  # Plot every 10th state
            probability = np.abs(state)**2
            ax1.plot(x_grid, probability, alpha=0.5, label=f't={t:.2e}')
            
        ax1.set_title("Wavefunction Evolution")
        ax1.set_xlabel("Position")
        ax1.set_ylabel("Probability Density")
        ax1.legend()
        
        # Plot coherence decay
        coherence = [np.abs(np.vdot(state, state)) for state in states]
        theoretical = np.exp(-INFORMATION_GENERATION_RATE * np.array(times))
        
        ax2.plot(times, coherence, 'b-', label='Actual')
        ax2.plot(times, theoretical, 'r--', label='Theoretical')
        ax2.set_title("Coherence Decay")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Coherence")
        ax2.set_yscale('log')
        ax2.legend()
        
        if save:
            plt.savefig(self.output_dir / "state_evolution.png")
        plt.close()
    
    def plot_metrics_summary(
        self,
        metrics: List[StateMetrics],
        save: bool = True
    ) -> None:
        """Plot summary of simulation metrics."""
        times = [m.time for m in metrics]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Energy evolution
        axes[0,0].plot(times, [m.energy for m in metrics])
        axes[0,0].set_title("Energy Evolution")
        axes[0,0].set_xlabel("Time")
        axes[0,0].set_ylabel("Energy")
        
        # Information content
        axes[0,1].plot(times, [m.information_content for m in metrics])
        axes[0,1].set_title("Information Content")
        axes[0,1].set_xlabel("Time")
        axes[0,1].set_ylabel("Information (bits)")
        
        # Integration measure
        axes[1,0].plot(times, [m.integration_measure for m in metrics])
        axes[1,0].set_title("Integration Measure (Φ)")
        axes[1,0].set_xlabel("Time")
        axes[1,0].set_ylabel("Φ")
        
        # Entropy
        axes[1,1].plot(times, [m.entropy for m in metrics])
        axes[1,1].set_title("Entropy Evolution")
        axes[1,1].set_xlabel("Time")
        axes[1,1].set_ylabel("S")
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "metrics_summary.png")
        plt.close() 