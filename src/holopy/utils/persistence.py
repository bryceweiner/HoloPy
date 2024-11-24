"""
State persistence system implementing holographic serialization and recovery.
"""
import h5py
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import asdict
import zlib
import lz4.frame
import blosc
from enum import Enum
from ..config.constants import INFORMATION_GENERATION_RATE

logger = logging.getLogger(__name__)

class CompressionMethod(Enum):
    """Available compression methods."""
    ZLIB = "zlib"
    LZ4 = "lz4"
    BLOSC = "blosc"

class StatePersistence:
    """Manages state persistence with holographic constraints."""
    
    def __init__(
        self,
        base_path: Path,
        compression_method: CompressionMethod = CompressionMethod.BLOSC,
        compression_level: int = 6,
        max_checkpoints: int = 5
    ):
        self.base_path = Path(base_path)
        self.compression_method = compression_method
        self.compression_level = compression_level
        self.max_checkpoints = max_checkpoints
        
        # Create directory structure
        self.states_path = self.base_path / "states"
        self.metadata_path = self.base_path / "metadata"
        self.checkpoints_path = self.base_path / "checkpoints"
        self.versions_path = self.base_path / "versions"
        
        for path in [
            self.states_path,
            self.metadata_path,
            self.checkpoints_path,
            self.versions_path
        ]:
            path.mkdir(parents=True, exist_ok=True)
            
        # Initialize subsystems
        self.version_control = VersionControl(self.versions_path)
        self.recovery_system = RecoverySystem(
            self.base_path,
            self.version_control,
            self
        )
        
        logger.info(
            f"Initialized StatePersistence at {base_path} "
            f"with compression_method={compression_method.value}, "
            f"compression_level={compression_level}"
        )

    def save_state(
        self,
        state: np.ndarray,
        metadata: Dict[str, Any],
        timestamp: float,
        is_checkpoint: bool = False
    ) -> Path:
        """Save quantum state with version control and compression."""
        # Apply holographic corrections
        corrected_state = self._apply_holographic_corrections(state, timestamp)
        
        # Compress state data
        compressed_data, compression_info = self._compress_data(corrected_state)
        
        # Create version entry
        version = self.version_control.create_version(
            compressed_data,
            {**metadata, "compression": compression_info},
            compression_level=self.compression_level
        )
        
        # Save state data
        state_path = self.states_path / f"state_{version.version_id}.h5"
        with h5py.File(state_path, 'w') as f:
            f.create_dataset('state', data=compressed_data)
            f.attrs['version_id'] = version.version_id
            f.attrs['timestamp'] = timestamp
        
        # Create checkpoint if requested
        if is_checkpoint:
            self._create_checkpoint(state_path, version.version_id)
        
        return state_path

    def load_state(
        self,
        version_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        latest_checkpoint: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Load quantum state with recovery mechanisms."""
        try:
            if latest_checkpoint:
                state_path = self._get_latest_checkpoint()
                if not state_path:
                    raise ValueError("No checkpoints found")
            elif version_id:
                state_path = self.states_path / f"state_{version_id}.h5"
            elif timestamp is not None:
                state_path = self._find_state_by_timestamp(timestamp)
            else:
                raise ValueError("Must specify version_id, timestamp, or latest_checkpoint")

            # Attempt recovery if needed
            if not state_path.exists():
                return self.recovery_system.recover_state(version_id)

            with h5py.File(state_path, 'r') as f:
                compressed_data = f['state'][:]
                version_id = f.attrs['version_id']
                timestamp = f.attrs['timestamp']

            # Verify version
            if not self.version_control.verify_version(version_id, compressed_data):
                logger.warning(f"Version verification failed for {version_id}")
                return self.recovery_system.recover_state(version_id)

            # Decompress and reconstruct state
            state = self._decompress_data(compressed_data)
            state = self._reconstruct_holographic_state(state, timestamp)

            # Load metadata
            metadata = self._load_metadata(version_id)

            return state, metadata

        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return None, None

    def _compress_data(self, data: np.ndarray) -> Tuple[bytes, Dict]:
        """Compress state data using selected method."""
        state_bytes = data.tobytes()
        
        if self.compression_method == CompressionMethod.ZLIB:
            compressed = zlib.compress(state_bytes, self.compression_level)
        elif self.compression_method == CompressionMethod.LZ4:
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
            "method": self.compression_method.value,
            "original_size": len(state_bytes),
            "compressed_size": len(compressed),
            "dtype": str(data.dtype),
            "shape": data.shape
        }
        
        return compressed, compression_info
    
    def _apply_holographic_corrections(
        self,
        state: np.ndarray,
        timestamp: float
    ) -> np.ndarray:
        """Apply holographic corrections before saving."""
        # Implementation based on equation from math.tex:2763-2765
        return state * np.exp(-INFORMATION_GENERATION_RATE * timestamp / 2)
    
    def _reconstruct_holographic_state(
        self,
        state: np.ndarray,
        timestamp: float
    ) -> np.ndarray:
        """Reconstruct state with holographic corrections."""
        # Implementation based on equation from math.tex:2757-2759
        return state * np.exp(INFORMATION_GENERATION_RATE * timestamp / 2)
    
    def _decompress_data(self, compressed_data: np.ndarray) -> np.ndarray:
        """Decompress state data."""
        return np.frombuffer(
            zlib.decompress(compressed_data.tobytes()),
            dtype=np.complex128
        )
    
    def _get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint file."""
        checkpoints = sorted(self.checkpoints_path.glob("state_*.h5"))
        return checkpoints[-1] if checkpoints else None
    
    def _find_state_file(
        self,
        timestamp: Optional[float]
    ) -> Optional[Path]:
        """Find appropriate state file for given timestamp."""
        if timestamp is None:
            # Get latest state
            states = sorted(self.states_path.glob("state_*.h5"))
            return states[-1] if states else None
            
        # Find closest state to timestamp
        states = sorted(
            self.states_path.glob("state_*.h5"),
            key=lambda p: abs(float(p.stem.split('_')[1]) - timestamp)
        )
        return states[0] if states else None
    
    def _manage_checkpoints(self) -> None:
        """Manage number of checkpoints."""
        checkpoints = sorted(self.checkpoints_path.glob("state_*.h5"))
        if len(checkpoints) > self.max_checkpoints:
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-self.max_checkpoints]:
                checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint}") 