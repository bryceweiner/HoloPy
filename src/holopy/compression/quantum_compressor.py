from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.sparse.linalg import svds
from dataclasses import dataclass
import torch
from ..config.constants import (
    COMPRESSION_THRESHOLD,
    MAX_BOND_DIMENSION,
    SVD_CUTOFF
)
import logging

logger = logging.getLogger(__name__)

@dataclass
class CompressionResult:
    """Contains compression results and metrics."""
    compressed_state: np.ndarray
    compression_ratio: float
    information_loss: float
    entanglement_preserved: float
    bond_dimensions: List[int]

class QuantumCompressor:
    """Advanced quantum state compression system."""
    
    def __init__(
        self,
        spatial_points: int,
        spatial_extent: float,
        enable_adaptive: bool = True
    ):
        self.spatial_points = spatial_points
        self.spatial_extent = spatial_extent
        self.enable_adaptive = enable_adaptive
        
        # Initialize tensor network structure
        self.bond_dimensions = [MAX_BOND_DIMENSION] * (spatial_points - 1)
        
        # Track compression history
        self.compression_history: List[CompressionResult] = []
        
        logger.info(f"Initialized QuantumCompressor")
    
    def compress_state(
        self,
        quantum_state: np.ndarray,
        target_size: Optional[int] = None
    ) -> CompressionResult:
        """Compress quantum state using tensor network decomposition."""
        try:
            # Calculate initial size
            initial_size = quantum_state.size
            
            # Reshape state for tensor decomposition
            matrix_dim = int(np.sqrt(initial_size))
            state_matrix = quantum_state.reshape(matrix_dim, -1)
            
            # Perform SVD compression
            compressed_state, bond_dims = self._tensor_decomposition(
                state_matrix,
                target_size
            )
            
            # Calculate compression metrics
            compression_ratio = compressed_state.size / initial_size
            information_loss = self._calculate_information_loss(
                quantum_state,
                compressed_state.flatten()
            )
            entanglement = self._calculate_entanglement_preservation(
                quantum_state,
                compressed_state.flatten()
            )
            
            # Create result
            result = CompressionResult(
                compressed_state=compressed_state.flatten(),
                compression_ratio=compression_ratio,
                information_loss=information_loss,
                entanglement_preserved=entanglement,
                bond_dimensions=bond_dims
            )
            
            # Update history
            self.compression_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"State compression failed: {str(e)}")
            raise
    
    def _tensor_decomposition(
        self,
        state_matrix: np.ndarray,
        target_size: Optional[int]
    ) -> Tuple[np.ndarray, List[int]]:
        """Perform tensor network decomposition."""
        try:
            # Calculate adaptive SVD cutoff if enabled
            if self.enable_adaptive and target_size is not None:
                cutoff = self._calculate_adaptive_cutoff(
                    state_matrix,
                    target_size
                )
            else:
                cutoff = SVD_CUTOFF
            
            # Perform SVD
            U, s, Vh = svds(state_matrix, k=min(
                state_matrix.shape[0]-1,
                MAX_BOND_DIMENSION
            ))
            
            # Apply cutoff
            mask = s > cutoff
            U = U[:, mask]
            s = s[mask]
            Vh = Vh[mask, :]
            
            # Update bond dimensions
            bond_dims = [U.shape[1]] * (self.spatial_points - 1)
            
            # Reconstruct compressed state
            compressed = U @ np.diag(s) @ Vh
            
            return compressed, bond_dims
            
        except Exception as e:
            logger.error(f"Tensor decomposition failed: {str(e)}")
            raise
    
    def _calculate_adaptive_cutoff(
        self,
        state_matrix: np.ndarray,
        target_size: int
    ) -> float:
        """Calculate adaptive SVD cutoff for target compression."""
        try:
            # Perform full SVD
            U, s, Vh = np.linalg.svd(state_matrix, full_matrices=False)
            
            # Calculate cumulative explained variance
            explained_variance = np.cumsum(s**2) / np.sum(s**2)
            
            # Find cutoff that achieves target size
            target_variance = 1.0 - (target_size / state_matrix.size)
            cutoff_idx = np.searchsorted(explained_variance, target_variance)
            
            if cutoff_idx < len(s):
                return s[cutoff_idx]
            else:
                return SVD_CUTOFF
                
        except Exception as e:
            logger.error(f"Adaptive cutoff calculation failed: {str(e)}")
            raise
    
    def _calculate_information_loss(
        self,
        original_state: np.ndarray,
        compressed_state: np.ndarray
    ) -> float:
        """Calculate information loss from compression."""
        try:
            # Calculate quantum relative entropy
            p = np.abs(original_state)**2
            q = np.abs(compressed_state)**2
            
            # Add small constant for numerical stability
            eps = 1e-10
            relative_entropy = np.sum(
                p * np.log2((p + eps) / (q + eps))
            )
            
            return float(relative_entropy)
            
        except Exception as e:
            logger.error(f"Information loss calculation failed: {str(e)}")
            raise
    
    def _calculate_entanglement_preservation(
        self,
        original_state: np.ndarray,
        compressed_state: np.ndarray
    ) -> float:
        """Calculate preserved entanglement after compression."""
        try:
            # Calculate entanglement entropy for both states
            original_entropy = self._von_neumann_entropy(
                original_state.reshape(-1, 1)
            )
            compressed_entropy = self._von_neumann_entropy(
                compressed_state.reshape(-1, 1)
            )
            
            # Calculate preservation ratio
            if original_entropy > 0:
                preservation = compressed_entropy / original_entropy
            else:
                preservation = 1.0
            
            return float(preservation)
            
        except Exception as e:
            logger.error(f"Entanglement preservation calculation failed: {str(e)}")
            raise
    
    def _von_neumann_entropy(
        self,
        state: np.ndarray
    ) -> float:
        """Calculate von Neumann entropy."""
        try:
            # Calculate density matrix
            rho = state @ state.conj().T
            
            # Calculate eigenvalues
            eigvals = np.linalg.eigvalsh(rho)
            
            # Calculate entropy
            entropy = -np.sum(
                eigvals * np.log2(eigvals + 1e-10)
            )
            
            return float(entropy)
            
        except Exception as e:
            logger.error(f"Entropy calculation failed: {str(e)}")
            raise 