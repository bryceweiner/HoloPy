"""
Advanced compression system for quantum states.
"""
import numpy as np
from typing import Tuple, Optional
import zlib
import lz4.frame
import blosc
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CompressionMethod(Enum):
    """Available compression methods."""
    ZLIB = "zlib"
    LZ4 = "lz4"
    BLOSC = "blosc"

class CompressionManager:
    """Manages compression of quantum states."""
    
    def __init__(
        self,
        default_method: CompressionMethod = CompressionMethod.BLOSC,
        compression_level: int = 6
    ):
        self.default_method = default_method
        self.compression_level = compression_level
        
    def compress_state(
        self,
        state: np.ndarray,
        method: Optional[CompressionMethod] = None
    ) -> Tuple[bytes, Dict]:
        """Compress quantum state data."""
        method = method or self.default_method
        
        # Convert to bytes
        state_bytes = state.tobytes()
        
        # Apply compression
        if method == CompressionMethod.ZLIB:
            compressed = zlib.compress(state_bytes, self.compression_level)
        elif method == CompressionMethod.LZ4:
            compressed = lz4.frame.compress(
                state_bytes,
                compression_level=self.compression_level
            )
        else:  # BLOSC
            compressed = blosc.compress(
                state_bytes,
                typesize=16,  # complex128
                clevel=self.compression_level
            )
            
        compression_info = {
            "method": method.value,
            "original_size": len(state_bytes),
            "compressed_size": len(compressed),
            "dtype": str(state.dtype),
            "shape": state.shape
        }
        
        return compressed, compression_info 