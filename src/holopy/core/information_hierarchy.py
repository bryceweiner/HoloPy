from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from ..config.constants import (
    INFORMATION_GENERATION_RATE,
    COUPLING_CONSTANT,
    PLANCK_LENGTH,
    LIGHT_SPEED
)
import logging

logger = logging.getLogger(__name__)

@dataclass
class HierarchyLevel:
    """Represents a level in the information processing hierarchy."""
    level: int
    processing_rate: float
    coherence_length: float
    coupling_strength: float
    active: bool = True

class InformationHierarchyProcessor:
    """Manages multi-level information processing in holographic framework."""
    
    def __init__(
        self,
        spatial_points: int,
        spatial_extent: float,
        num_levels: int = 3
    ):
        self.spatial_points = spatial_points
        self.spatial_extent = spatial_extent
        self.num_levels = num_levels
        
        # Initialize hierarchy levels
        self.levels = self._initialize_hierarchy()
        
        # Metrics tracking
        self.hierarchy_metrics: List[Dict[str, float]] = []
        
        logger.info(f"Initialized {num_levels}-level information hierarchy")
    
    def _initialize_hierarchy(self) -> List[HierarchyLevel]:
        """Initialize information processing hierarchy levels."""
        try:
            levels = []
            base_rate = INFORMATION_GENERATION_RATE
            
            for i in range(self.num_levels):
                # Each level has progressively slower processing rate
                level_rate = base_rate / (2**i)
                
                # Coherence length increases with level
                coherence_length = PLANCK_LENGTH * np.exp(i)
                
                # Coupling strength decreases with level
                coupling = COUPLING_CONSTANT / (i + 1)
                
                levels.append(HierarchyLevel(
                    level=i,
                    processing_rate=level_rate,
                    coherence_length=coherence_length,
                    coupling_strength=coupling
                ))
            
            return levels
            
        except Exception as e:
            logger.error(f"Failed to initialize hierarchy: {str(e)}")
            raise
    
    def process_state(
        self,
        wavefunction: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Process wavefunction through information hierarchy."""
        try:
            processed = wavefunction.copy()
            metrics = {}
            
            # Process through each active level
            for level in self.levels:
                if not level.active:
                    continue
                    
                # Apply level-specific processing
                processed = self._apply_level_processing(
                    processed,
                    level,
                    dt
                )
                
                # Calculate level metrics
                level_metrics = self._calculate_level_metrics(
                    processed,
                    level
                )
                
                metrics[f"level_{level.level}"] = level_metrics
                
                # Check coherence threshold for level deactivation
                if level_metrics['coherence'] < 0.1:
                    level.active = False
                    logger.info(f"Deactivated hierarchy level {level.level}")
            
            # Update hierarchy metrics
            self.hierarchy_metrics.append(metrics)
            
            return processed, metrics
            
        except Exception as e:
            logger.error(f"State processing failed: {str(e)}")
            raise
    
    def _apply_level_processing(
        self,
        wavefunction: np.ndarray,
        level: HierarchyLevel,
        dt: float
    ) -> np.ndarray:
        """Apply level-specific information processing."""
        try:
            # Calculate coherence kernel
            x = np.linspace(
                -self.spatial_extent/2,
                self.spatial_extent/2,
                self.spatial_points
            )
            dx = x[1] - x[0]
            
            # Level-specific coherence kernel
            kernel = np.exp(
                -(x[:, np.newaxis] - x[np.newaxis, :])**2 /
                (2 * level.coherence_length**2)
            )
            
            # Apply processing rate decay
            processed = wavefunction * np.exp(-level.processing_rate * dt)
            
            # Apply coherence effects
            processed = np.dot(kernel, processed) / self.spatial_points
            
            # Normalize
            processed /= np.sqrt(np.sum(np.abs(processed)**2))
            
            return processed
            
        except Exception as e:
            logger.error(f"Level processing failed: {str(e)}")
            raise
    
    def _calculate_level_metrics(
        self,
        wavefunction: np.ndarray,
        level: HierarchyLevel
    ) -> Dict[str, float]:
        """Calculate metrics for hierarchy level."""
        try:
            density = np.abs(wavefunction)**2
            
            metrics = {
                'coherence': np.abs(np.sum(wavefunction * np.conj(wavefunction))),
                'information_content': -np.sum(density * np.log2(density + 1e-10)),
                'processing_rate': level.processing_rate,
                'coupling_strength': level.coupling_strength,
                'active': level.active
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {str(e)}")
            raise 