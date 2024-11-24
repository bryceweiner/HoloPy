"""
Integration tests for active inference and error correction system.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from holopy.inference.active_inference import ActiveInferenceEngine
from holopy.core.hilbert_continuum import HilbertContinuum
from holopy.metrics.collectors import MetricsCollector
from holopy.config.constants import (
    INFORMATION_GENERATION_RATE,
    COUPLING_CONSTANT
)

@pytest.fixture
def setup_inference_system():
    """Set up complete inference testing system."""
    spatial_points = 128
    dt = 0.01
    total_time = 1.0
    
    # Initialize components
    inference_engine = ActiveInferenceEngine(
        spatial_points=spatial_points,
        dt=dt,
        learning_rate=0.01,
        prediction_horizon=10
    )
    
    hilbert = HilbertContinuum(
        spatial_points=spatial_points,
        spatial_extent=10.0
    )
    
    metrics = MetricsCollector()
    
    return inference_engine, hilbert, metrics, dt, total_time

def test_state_prediction_accuracy(setup_inference_system):
    """Test prediction accuracy with holographic constraints."""
    inference_engine, hilbert, metrics, dt, total_time = setup_inference_system
    
    # Create initial state
    hilbert.create_initial_state()
    initial_state = hilbert.matter_wavefunction.copy()
    
    # Generate prediction
    predicted_state, pred_metrics = inference_engine.predict_state(
        initial_state,
        time_steps=5
    )
    
    # Evolve actual state
    for _ in range(5):
        hilbert.evolve(dt)
    
    actual_state = hilbert.matter_wavefunction
    
    # Verify prediction accuracy
    error = np.mean(np.abs(predicted_state - actual_state))
    assert error < 0.1, f"Prediction error {error} exceeds threshold"
    
    # Verify holographic constraints
    density = np.abs(predicted_state)**2
    entropy = -np.sum(density * np.log(density + 1e-10))
    assert entropy <= len(predicted_state), "Holographic bound violated"
    
    # Check information conservation
    assert pred_metrics.information_gain >= 0, "Negative information gain"
    assert pred_metrics.processing_cost > 0, "Invalid processing cost"

def test_error_correction_convergence(setup_inference_system):
    """Test error correction convergence with active inference."""
    inference_engine, hilbert, metrics, dt, total_time = setup_inference_system
    
    # Initialize system
    hilbert.create_initial_state()
    
    # Evolution with error correction
    times = np.arange(0, total_time, dt)
    correction_history = []
    
    for t in times:
        # Generate prediction
        predicted_state, _ = inference_engine.predict_state(
            hilbert.matter_wavefunction,
            time_steps=1
        )
        
        # Evolve actual state
        hilbert.evolve(dt)
        
        # Apply correction
        corrected_state, confidence = inference_engine.correct_state(
            predicted_state,
            hilbert.matter_wavefunction
        )
        
        # Update state and track metrics
        hilbert.matter_wavefunction = corrected_state
        correction_history.append(confidence)
        
        # Collect metrics
        metrics.collect_state_metrics(
            corrected_state,
            t,
            {'confidence': confidence}
        )
    
    # Verify correction convergence
    confidence_trend = np.array(correction_history)
    assert np.mean(confidence_trend[-10:]) > np.mean(confidence_trend[:10]), \
        "Correction confidence did not improve"
    
    # Validate metrics
    df = metrics.tracking_df
    
    # Check information conservation with corrections
    initial_info = df.iloc[0].information_content
    final_info = df.iloc[-1].information_content
    expected_info = initial_info * np.exp(-INFORMATION_GENERATION_RATE * total_time)
    assert np.abs(final_info - expected_info) < 1e-6, \
        "Information conservation violated"

def test_active_inference_adaptation(setup_inference_system):
    """Test active inference model adaptation to system dynamics."""
    inference_engine, hilbert, metrics, dt, total_time = setup_inference_system
    
    # Initialize with perturbed coupling
    hilbert.create_initial_state()
    original_coupling = COUPLING_CONSTANT
    perturbed_coupling = COUPLING_CONSTANT * 1.2
    
    prediction_errors = []
    adaptation_metrics = []
    
    # Evolution with changing dynamics
    times = np.arange(0, total_time, dt)
    mid_point = len(times) // 2
    
    for i, t in enumerate(times):
        # Change coupling halfway through
        if i == mid_point:
            inference_engine.model_params['coupling_strength'] = perturbed_coupling
        
        # Generate prediction
        predicted_state, pred_metrics = inference_engine.predict_state(
            hilbert.matter_wavefunction,
            time_steps=1
        )
        
        # Evolve and correct
        hilbert.evolve(dt)
        corrected_state, confidence = inference_engine.correct_state(
            predicted_state,
            hilbert.matter_wavefunction
        )
        
        # Track metrics
        prediction_errors.append(pred_metrics.prediction_error)
        adaptation_metrics.append({
            'time': t,
            'error': pred_metrics.prediction_error,
            'confidence': confidence,
            'information_gain': pred_metrics.information_gain
        })
        
        hilbert.matter_wavefunction = corrected_state
    
    # Convert metrics to DataFrame
    adaptation_df = pd.DataFrame(adaptation_metrics)
    
    # Verify adaptation to changed dynamics
    pre_change_error = np.mean(prediction_errors[:mid_point])
    post_change_error = np.mean(prediction_errors[-mid_point:])
    
    assert post_change_error < pre_change_error * 1.5, \
        "Failed to adapt to changed dynamics"
    
    # Verify information processing efficiency
    information_efficiency = adaptation_df.information_gain / \
                           adaptation_df.error
    
    assert np.mean(information_efficiency[-mid_point:]) > \
           np.mean(information_efficiency[:mid_point]), \
           "Information processing efficiency did not improve" 