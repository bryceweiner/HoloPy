"""
Constants module for the holographic universe simulation.
Contains fundamental physical constants and derived parameters.
"""
from dataclasses import dataclass
import numpy as np

# Fundamental Constants
INFORMATION_GENERATION_RATE = 1.89e-29  # γ in s⁻¹
PLANCK_CONSTANT = 6.62607015e-34  # ℏ in J⋅s
SPEED_OF_LIGHT = 2.99792458e8  # c in m/s
BOLTZMANN_CONSTANT = 1.380649e-23  # k in J/K
GRAVITATIONAL_CONSTANT = 6.67430e-11  # G in m³/kg⋅s²

# Derived Constants
MATTER_RATE = INFORMATION_GENERATION_RATE / 2  # γ_m
ANTIMATTER_RATE = INFORMATION_GENERATION_RATE / 2  # γ_a

# Processing Hierarchy Rates
GAMMA_1 = INFORMATION_GENERATION_RATE / (2 * np.pi)  # Primary rate
GAMMA_2 = None  # Will be calculated based on entropy
GAMMA_3 = None  # Will be calculated based on temperature

# Integration Measures
CRITICAL_THRESHOLD = GAMMA_1 * PLANCK_CONSTANT * np.log(2)  # Consciousness threshold

# Field Propagator Constants
COUPLING_CONSTANT = 1 / (4 * np.pi * 256)  # α 