"""
Version control system for state persistence.
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Optional, List
import hashlib
import logging

logger = logging.getLogger(__name__)

@dataclass
class StateVersion:
    """Version information for saved states."""
    version_id: str
    timestamp: float
    parent_id: Optional[str]
    metadata: Dict
    checksum: str
    compression_level: int

class VersionControl:
    """Manages versioning for saved quantum states."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.versions_path = base_path / "versions"
        self.versions_path.mkdir(parents=True, exist_ok=True)
        self._load_version_history()
        
    def create_version(
        self,
        state_data: bytes,
        metadata: Dict,
        parent_id: Optional[str] = None,
        compression_level: int = 6
    ) -> StateVersion:
        """Create new version entry for state."""
        version_id = self._generate_version_id()
        checksum = self._calculate_checksum(state_data)
        
        version = StateVersion(
            version_id=version_id,
            timestamp=datetime.now().timestamp(),
            parent_id=parent_id,
            metadata=metadata,
            checksum=checksum,
            compression_level=compression_level
        )
        
        self._save_version_info(version)
        return version
    
    def verify_version(self, version_id: str, state_data: bytes) -> bool:
        """Verify state data matches version checksum."""
        version = self._load_version_info(version_id)
        if not version:
            return False
        
        current_checksum = self._calculate_checksum(state_data)
        return current_checksum == version.checksum
    
    def _generate_version_id(self) -> str:
        """Generate unique version identifier."""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:12]
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum of state data."""
        return hashlib.sha256(data).hexdigest() 