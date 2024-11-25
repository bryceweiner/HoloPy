import numpy as np
import pytest
from pathlib import Path
from holopy.metrics.collectors import MetricsCollector

def test_metrics_collector_initialization():
    output_dir = Path("test_output")
    collector = MetricsCollector(output_dir=output_dir)
    assert collector.output_dir == output_dir
    assert hasattr(collector, "metrics")

def test_metrics_collection(hilbert_continuum):
    hilbert_continuum.create_initial_state()
    metrics = hilbert_continuum.metrics_collector.metrics
    
    # Check basic metrics are being collected
    expected_metrics = {
        "time", "energy", "coherence", "coupling_strength",
        "temperature", "entropy", "information_content"
    }
    
    # Check that all expected metrics are present
    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
    
    # Check that metrics have valid values
    for metric_name, values in metrics.items():
        assert len(values) > 0, f"No values collected for {metric_name}"
        assert all(np.isfinite(v) for v in values), f"Invalid values in {metric_name}"