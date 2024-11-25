"""
Basic usage example of HoloPy framework.
"""

import numpy as np
from holopy import HilbertSpace, HilbertContinuum
import matplotlib.pyplot as plt

def main():
    # Initialize the space
    hilbert_space = HilbertSpace(dimension=128, extent=10.0)
    
    # Create simulation
    simulation = HilbertContinuum(
        hilbert_space=hilbert_space,
        dt=0.01,
        enable_hierarchy=True
    )
    
    # Run simulation
    simulation.create_initial_state()
    simulation.evolve(steps=100)
        
    # Plot results
    metrics = simulation.metrics_df
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['time'], metrics['energy'], label='Energy')
    plt.plot(metrics['time'], metrics['entropy'], label='Entropy')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title('System Evolution')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main() 