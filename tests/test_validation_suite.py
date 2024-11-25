import pytest
import numpy as np
import pandas as pd
from holopy.metrics.validation_suite import ValidationResult

def test_validation_result_creation():
    result = ValidationResult(
        passed=True,
        error=0.001,
        message="Test passed",
        details={"value": 0.5}
    )
    assert result.passed
    assert result.error == 0.001
    assert result.message == "Test passed"
    assert result.details["value"] == 0.5

def test_information_bound_validation(validation_suite, sample_state):
    metrics_df = pd.DataFrame({
        "time": [0.0],
        "energy": [1.0],
        "coherence": [1.0]
    })
    
    result = validation_suite._validate_information_bound(sample_state, metrics_df)
    assert isinstance(result, ValidationResult)
    assert result.passed
    assert result.error >= 0

def test_energy_conservation(validation_suite):
    metrics_df = pd.DataFrame({
        "time": [0.0, 0.1, 0.2],
        "energy": [1.0, 0.9999, 0.9998],
        "coherence": [1.0, 0.9999, 0.9998]
    })
    
    result = validation_suite._validate_energy_conservation(metrics_df, dt=0.1)
    assert isinstance(result, ValidationResult)
    assert result.passed
    assert "initial_energy" in result.details
    assert "current_energy" in result.details
    assert "expected_energy" in result.details
    assert "relative_error" in result.details
    
    assert result.details["initial_energy"] == 1.0
    assert result.details["current_energy"] == 0.9998
    assert result.error < validation_suite.tolerance

def test_antimatter_coherence(validation_suite, sample_state):
    result = validation_suite._validate_antimatter_coherence(sample_state)
    assert isinstance(result, ValidationResult)
    assert result.passed
    assert abs(result.details["norm"] - 1.0) < validation_suite.tolerance 