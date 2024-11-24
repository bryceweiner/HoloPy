from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from ..config.constants import (
    PLOT_STYLE,
    COLOR_PALETTE,
    ANIMATION_INTERVAL
)
import logging

logger = logging.getLogger(__name__)

class QuantumStateVisualizer:
    """Advanced visualization system for quantum state evolution."""
    
    def __init__(
        self,
        spatial_points: int,
        spatial_extent: float,
        interactive: bool = True
    ):
        self.spatial_points = spatial_points
        self.spatial_extent = spatial_extent
        self.interactive = interactive
        
        # Set style
        plt.style.use(PLOT_STYLE)
        sns.set_palette(COLOR_PALETTE)
        
        # Initialize plotly for interactive plots
        if interactive:
            self.fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Wavefunction Evolution',
                    'Phase Space',
                    'Information Flow',
                    'Entanglement Network'
                )
            )
        
        logger.info(f"Initialized QuantumStateVisualizer")
    
    def create_evolution_animation(
        self,
        states: List[Tuple[np.ndarray, np.ndarray]],
        times: np.ndarray
    ) -> FuncAnimation:
        """Create animation of quantum state evolution."""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            x = np.linspace(-self.spatial_extent/2, self.spatial_extent/2, self.spatial_points)
            
            def update(frame):
                matter, antimatter = states[frame]
                
                # Clear axes
                ax1.clear()
                ax2.clear()
                
                # Plot matter wavefunction
                ax1.plot(x, np.abs(matter)**2, 'b-', label='Matter')
                ax1.set_title(f'Time: {times[frame]:.3f}')
                ax1.set_ylabel('Probability Density')
                ax1.legend()
                
                # Plot antimatter wavefunction
                ax2.plot(x, np.abs(antimatter)**2, 'r-', label='Antimatter')
                ax2.set_xlabel('Position')
                ax2.set_ylabel('Probability Density')
                ax2.legend()
            
            anim = FuncAnimation(
                fig,
                update,
                frames=len(states),
                interval=ANIMATION_INTERVAL
            )
            
            return anim
            
        except Exception as e:
            logger.error(f"Animation creation failed: {str(e)}")
            raise
    
    def plot_phase_space_trajectory(
        self,
        metrics_df: pd.DataFrame,
        highlight_regimes: bool = True
    ) -> go.Figure:
        """Create interactive phase space visualization."""
        try:
            if self.interactive:
                # Create phase space scatter plot
                self.fig.add_trace(
                    go.Scatter(
                        x=metrics_df['energy'],
                        y=metrics_df['information_content'],
                        mode='lines+markers',
                        marker=dict(
                            color=metrics_df['time'],
                            colorscale='Viridis',
                            showscale=True
                        ),
                        name='Phase Space Trajectory'
                    ),
                    row=1, col=2
                )
                
                if highlight_regimes:
                    # Add regime change points
                    changes = self._detect_regime_changes(metrics_df)
                    for change in changes:
                        self.fig.add_vline(
                            x=metrics_df['energy'].iloc[change],
                            line_dash="dash",
                            line_color="red",
                            row=1, col=2
                        )
                
                return self.fig
            else:
                logger.warning("Interactive mode disabled")
                return None
                
        except Exception as e:
            logger.error(f"Phase space visualization failed: {str(e)}")
            raise
    
    def plot_entanglement_network(
        self,
        matter_wavefunction: np.ndarray,
        antimatter_wavefunction: np.ndarray,
        threshold: float = 0.1
    ) -> go.Figure:
        """Visualize quantum entanglement network."""
        try:
            if self.interactive:
                # Calculate density matrix
                rho = np.outer(matter_wavefunction, np.conj(antimatter_wavefunction))
                
                # Create network
                G = nx.Graph()
                for i in range(len(rho)):
                    for j in range(len(rho)):
                        if abs(rho[i,j]) > threshold:
                            G.add_edge(i, j, weight=abs(rho[i,j]))
                
                # Get network layout
                pos = nx.spring_layout(G)
                
                # Create edge trace
                edge_x = []
                edge_y = []
                edge_weights = []
                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_weights.extend([edge[2]['weight'], edge[2]['weight'], None])
                
                self.fig.add_trace(
                    go.Scatter(
                        x=edge_x,
                        y=edge_y,
                        mode='lines',
                        line=dict(
                            width=1,
                            color=edge_weights,
                            colorscale='Viridis'
                        ),
                        hoverinfo='none'
                    ),
                    row=2, col=2
                )
                
                return self.fig
            else:
                logger.warning("Interactive mode disabled")
                return None
                
        except Exception as e:
            logger.error(f"Entanglement network visualization failed: {str(e)}")
            raise
    
    def plot_information_flow(
        self,
        metrics_df: pd.DataFrame
    ) -> go.Figure:
        """Visualize information flow patterns."""
        try:
            if self.interactive:
                # Calculate flow rates
                flow_rates = np.gradient(
                    metrics_df['information_content'].values,
                    metrics_df['time'].values
                )
                
                # Add flow rate trace
                self.fig.add_trace(
                    go.Scatter(
                        x=metrics_df['time'],
                        y=flow_rates,
                        mode='lines',
                        name='Information Flow Rate'
                    ),
                    row=2, col=1
                )
                
                # Add flow acceleration
                flow_accel = np.gradient(flow_rates, metrics_df['time'].values)
                self.fig.add_trace(
                    go.Scatter(
                        x=metrics_df['time'],
                        y=flow_accel,
                        mode='lines',
                        name='Flow Acceleration',
                        line=dict(dash='dash')
                    ),
                    row=2, col=1
                )
                
                return self.fig
            else:
                logger.warning("Interactive mode disabled")
                return None
                
        except Exception as e:
            logger.error(f"Information flow visualization failed: {str(e)}")
            raise
    
    def _detect_regime_changes(
        self,
        metrics_df: pd.DataFrame,
        threshold: float = 2.0
    ) -> List[int]:
        """Detect regime changes in metrics."""
        try:
            # Calculate combined signal
            signal = (
                metrics_df['energy'].values / metrics_df['energy'].std() +
                metrics_df['information_content'].values / metrics_df['information_content'].std()
            )
            
            # Detect changes using gradient
            gradient = np.gradient(signal)
            changes = np.where(abs(gradient) > threshold)[0]
            
            return changes
            
        except Exception as e:
            logger.error(f"Regime change detection failed: {str(e)}")
            raise 