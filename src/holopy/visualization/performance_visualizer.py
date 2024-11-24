from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
from ..benchmarks.system_benchmarks import BenchmarkResult
from ..config.constants import (
    PERFORMANCE_TARGETS,
    VISUALIZATION_CONFIG
)
import logging

logger = logging.getLogger(__name__)

class PerformanceVisualizer:
    """Visualization tools for system performance analysis."""
    
    def __init__(
        self,
        results_dir: Optional[Path] = None,
        interactive: bool = True
    ):
        self.results_dir = results_dir or Path("benchmark_results")
        self.interactive = interactive
        self.fig = None
        
    def create_performance_dashboard(
        self,
        benchmark_results: List[BenchmarkResult]
    ) -> go.Figure:
        """Create comprehensive performance dashboard."""
        try:
            # Create subplot figure
            self.fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Evolution Speed',
                    'Memory Usage',
                    'Scaling Analysis',
                    'Accuracy Metrics',
                    'Resource Utilization',
                    'Performance Overview'
                )
            )
            
            # Add evolution speed plot
            self._add_evolution_speed_plot(benchmark_results, row=1, col=1)
            
            # Add memory usage plot
            self._add_memory_usage_plot(benchmark_results, row=1, col=2)
            
            # Add scaling analysis
            self._add_scaling_analysis(benchmark_results, row=2, col=1)
            
            # Add accuracy metrics
            self._add_accuracy_metrics(benchmark_results, row=2, col=2)
            
            # Add resource utilization
            self._add_resource_utilization(benchmark_results, row=3, col=1)
            
            # Add performance overview
            self._add_performance_overview(benchmark_results, row=3, col=2)
            
            # Update layout
            self.fig.update_layout(
                height=1200,
                width=1600,
                showlegend=True,
                title_text="Holographic System Performance Dashboard"
            )
            
            return self.fig
            
        except Exception as e:
            logger.error(f"Dashboard creation failed: {str(e)}")
            raise
    
    def _add_evolution_speed_plot(
        self,
        results: List[BenchmarkResult],
        row: int,
        col: int
    ):
        """Add evolution speed analysis plot."""
        try:
            points = [r.spatial_points for r in results]
            speeds = [r.steps_per_second for r in results]
            targets = [
                PERFORMANCE_TARGETS[r.config_name]['min_steps_per_second']
                for r in results
            ]
            
            # Add actual speed trace
            self.fig.add_trace(
                go.Scatter(
                    x=points,
                    y=speeds,
                    mode='lines+markers',
                    name='Evolution Speed',
                    line=dict(color='blue')
                ),
                row=row, col=col
            )
            
            # Add target line
            self.fig.add_trace(
                go.Scatter(
                    x=points,
                    y=targets,
                    mode='lines',
                    name='Target Speed',
                    line=dict(color='red', dash='dash')
                ),
                row=row, col=col
            )
            
            self.fig.update_xaxes(
                title_text="Spatial Points",
                type="log",
                row=row, col=col
            )
            self.fig.update_yaxes(
                title_text="Steps per Second",
                type="log",
                row=row, col=col
            )
            
        except Exception as e:
            logger.error(f"Evolution speed plot failed: {str(e)}")
            raise
    
    def _add_memory_usage_plot(
        self,
        results: List[BenchmarkResult],
        row: int,
        col: int
    ):
        """Add memory usage analysis plot."""
        try:
            points = [r.spatial_points for r in results]
            memory = [r.memory_usage / 1e6 for r in results]  # Convert to MB
            
            # Create bar chart
            self.fig.add_trace(
                go.Bar(
                    x=points,
                    y=memory,
                    name='Memory Usage (MB)',
                    marker_color='green'
                ),
                row=row, col=col
            )
            
            self.fig.update_xaxes(
                title_text="Spatial Points",
                row=row, col=col
            )
            self.fig.update_yaxes(
                title_text="Memory Usage (MB)",
                row=row, col=col
            )
            
        except Exception as e:
            logger.error(f"Memory usage plot failed: {str(e)}")
            raise
    
    def _add_scaling_analysis(
        self,
        results: List[BenchmarkResult],
        row: int,
        col: int
    ):
        """Add computational scaling analysis plot."""
        try:
            points = np.array([r.spatial_points for r in results])
            times = np.array([r.total_time for r in results])
            
            # Fit scaling law
            log_points = np.log(points)
            log_times = np.log(times)
            slope, intercept = np.polyfit(log_points, log_times, 1)
            
            # Create scatter plot
            self.fig.add_trace(
                go.Scatter(
                    x=points,
                    y=times,
                    mode='markers',
                    name='Actual Times',
                    marker=dict(color='blue')
                ),
                row=row, col=col
            )
            
            # Add fit line
            fit_points = np.logspace(
                np.log10(points.min()),
                np.log10(points.max()),
                100
            )
            fit_times = np.exp(intercept + slope * np.log(fit_points))
            
            self.fig.add_trace(
                go.Scatter(
                    x=fit_points,
                    y=fit_times,
                    mode='lines',
                    name=f'Scaling: O(N^{slope:.2f})',
                    line=dict(color='red', dash='dash')
                ),
                row=row, col=col
            )
            
            self.fig.update_xaxes(
                title_text="Spatial Points",
                type="log",
                row=row, col=col
            )
            self.fig.update_yaxes(
                title_text="Computation Time (s)",
                type="log",
                row=row, col=col
            )
            
        except Exception as e:
            logger.error(f"Scaling analysis plot failed: {str(e)}")
            raise
    
    def _add_accuracy_metrics(
        self,
        results: List[BenchmarkResult],
        row: int,
        col: int
    ):
        """Add accuracy metrics analysis plot."""
        try:
            points = [r.spatial_points for r in results]
            info_errors = [
                r.accuracy_metrics['information_conservation_error']
                for r in results
            ]
            norm_errors = [
                r.accuracy_metrics['normalization_error']
                for r in results
            ]
            
            # Add information conservation errors
            self.fig.add_trace(
                go.Scatter(
                    x=points,
                    y=info_errors,
                    mode='lines+markers',
                    name='Information Conservation Error',
                    line=dict(color='blue')
                ),
                row=row, col=col
            )
            
            # Add normalization errors
            self.fig.add_trace(
                go.Scatter(
                    x=points,
                    y=norm_errors,
                    mode='lines+markers',
                    name='Normalization Error',
                    line=dict(color='red')
                ),
                row=row, col=col
            )
            
            self.fig.update_xaxes(
                title_text="Spatial Points",
                type="log",
                row=row, col=col
            )
            self.fig.update_yaxes(
                title_text="Error Magnitude",
                type="log",
                row=row, col=col
            )
            
        except Exception as e:
            logger.error(f"Accuracy metrics plot failed: {str(e)}")
            raise 