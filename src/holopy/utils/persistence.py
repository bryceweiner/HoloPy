"""
Persistence utilities for quantum states.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
from pathlib import Path
import zlib
import lzma
import bz2
from enum import Enum, auto
import json
from datetime import datetime
from .version_control import VersionControl
from ..config.constants import (
    COMPRESSION_LEVEL,
    COMPRESSION_METHODS,
    COMPRESSION_QUALITY_LEVELS,
    COMPRESSION_THRESHOLD,
    COUPLING_CONSTANT,
    CACHE_SIZE,
    VERSION_FORMAT
)
import logging

logger = logging.getLogger(__name__)

class CompressionMethod(Enum):
    """Available compression methods."""
    ZLIB = auto()
    LZMA = auto()
    BZIP2 = auto()
    WAVELET = auto()
    NONE = auto()

    @classmethod
    def get_compressor(cls, method: 'CompressionMethod'):
        """Get compression function for method."""
        compressors = {
            cls.ZLIB: zlib.compress,
            cls.LZMA: lzma.compress,
            cls.BZIP2: bz2.compress,
            cls.WAVELET: cls._wavelet_compress,
            cls.NONE: lambda x, **kwargs: x
        }
        return compressors[method]
    
    @classmethod
    def get_decompressor(cls, method: 'CompressionMethod'):
        """Get decompression function for method."""
        decompressors = {
            cls.ZLIB: zlib.decompress,
            cls.LZMA: lzma.decompress,
            cls.BZIP2: bz2.decompress,
            cls.WAVELET: cls._wavelet_decompress,
            cls.NONE: lambda x: x
        }
        return decompressors[method]
    
    @staticmethod
    def _wavelet_compress(data: bytes, level: int = 3) -> bytes:
        """Wavelet compression implementation."""
        try:
            # Convert bytes to numpy array
            arr = np.frombuffer(data, dtype=np.float64)
            
            # Perform wavelet transform
            coeffs = np.fft.fft(arr)
            
            # Threshold small coefficients
            threshold = np.max(np.abs(coeffs)) * 0.01 * level
            coeffs[np.abs(coeffs) < threshold] = 0
            
            # Pack coefficients
            compressed = {
                'coeffs': coeffs.tobytes(),
                'shape': arr.shape,
                'threshold': threshold
            }
            
            return json.dumps(compressed).encode()
            
        except Exception as e:
            logger.error(f"Wavelet compression failed: {str(e)}")
            raise
    
    @staticmethod
    def _wavelet_decompress(data: bytes) -> bytes:
        """Wavelet decompression implementation."""
        try:
            # Load compressed data
            compressed = json.loads(data.decode())
            
            # Unpack coefficients
            coeffs = np.frombuffer(
                compressed['coeffs'],
                dtype=np.complex128
            ).reshape(compressed['shape'])
            
            # Perform inverse transform
            arr = np.fft.ifft(coeffs)
            
            return arr.astype(np.float64).tobytes()
            
        except Exception as e:
            logger.error(f"Wavelet decompression failed: {str(e)}")
            raise
    
    @classmethod
    def from_string(cls, method_str: str) -> 'CompressionMethod':
        """Convert string to CompressionMethod."""
        try:
            return cls[method_str.upper()]
        except KeyError:
            raise ValueError(f"Unknown compression method: {method_str}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for compression method."""
        if self == self.NONE:
            return {'compression_type': 'none'}
        return COMPRESSION_METHODS[self.name.lower()]

class StatePersistence:
    """Handles persistence of quantum states with version control."""
    
    def __init__(self, storage_path: Path):
        """Initialize persistence system."""
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    def save_state(self, state: np.ndarray, metadata: Optional[Dict] = None, name: str = "state") -> Path:
        """Save quantum state with metadata."""
        if metadata is None:
            metadata = {}
            
        data = {
            'state': state,
            'metadata': metadata,
            'timestamp': np.datetime64('now')
        }
        
        path = self.storage_path / f"{name}.npy"
        np.save(path, data)
        return path
        
    def load_state(self, name: str = "state") -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load quantum state and metadata."""
        path = self.storage_path / f"{name}.npy"
        data = np.load(path, allow_pickle=True).item()
        return data['state'], data['metadata']