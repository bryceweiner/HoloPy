"""
Compression management system for holographic state data.
"""
from typing import Dict, Tuple, Optional, Union, Any
import numpy as np
import zlib
import logging

logger = logging.getLogger(__name__)

class CompressionManager:
    """Manages compression and decompression of quantum state data."""
    
    def __init__(self, compression_level: int = 6):
        """
        Initialize compression manager.
        
        Args:
            compression_level: Integer from 0-9 controlling compression level
        """
        self.compression_level = compression_level
        logger.info(f"Initialized CompressionManager with level {compression_level}")
    
    def compress_state(
        self,
        state_data: Union[np.ndarray, bytes],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compress quantum state data with metadata.
        
        Args:
            state_data: Numpy array or bytes to compress
            metadata: Optional dictionary of metadata
            
        Returns:
            Tuple of (compressed_bytes, metadata_dict)
        """
        try:
            # Convert numpy array to bytes if needed
            if isinstance(state_data, np.ndarray):
                state_bytes = state_data.tobytes()
            else:
                state_bytes = state_data
                
            # Compress the data
            compressed = zlib.compress(state_bytes, level=self.compression_level)
            
            # Update metadata
            meta = metadata or {}
            meta.update({
                'original_size': len(state_bytes),
                'compressed_size': len(compressed),
                'compression_ratio': len(compressed) / len(state_bytes)
            })
            
            return compressed, meta
            
        except Exception as e:
            logger.error(f"Compression failed: {str(e)}")
            raise