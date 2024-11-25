from dataclasses import dataclass


@dataclass
class PreparationMetrics:
    """Metrics for quantum state preparation."""
    preparation_fidelity: float = 0.0
    fidelity: float = 0.0
    purity: float = 0.0
    success_probability: float = 0.0
    preparation_time: float = 0.0
    gate_count: int = 0
    error_rate: float = 0.0
    entanglement_entropy: float = 0.0
    holographic_complexity: float = 0.0 