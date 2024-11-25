"""
Version control system for quantum states and configurations.
"""
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import shutil
import datetime
from dataclasses import dataclass
import numpy as np
from ..config.constants import VERSION_FORMAT
import logging

logger = logging.getLogger(__name__)

@dataclass
class Version:
    """Represents a version of a quantum state."""
    id: str
    timestamp: str
    description: str
    metadata: Dict[str, Any]
    parent_id: Optional[str] = None

class VersionControl:
    """Implements version control for quantum states."""
    
    def __init__(self, base_path: Path):
        """
        Initialize version control system.
        
        Args:
            base_path: Base directory for version storage
        """
        self.base_path = base_path
        self.versions_path = base_path / "versions"
        self.versions_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize version history
        self.versions: Dict[str, Version] = {}
        self.current_version: Optional[str] = None
        
        # Load existing versions
        self._load_version_history()
        
        logger.info(f"Initialized VersionControl at {base_path}")
    
    def _load_version_history(self) -> None:
        """Load version history from disk."""
        try:
            history_file = self.versions_path / "history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                # Load versions
                for version_data in history_data.get('versions', []):
                    version = Version(
                        id=version_data['id'],
                        timestamp=version_data['timestamp'],
                        description=version_data['description'],
                        metadata=version_data['metadata'],
                        parent_id=version_data.get('parent_id')
                    )
                    self.versions[version.id] = version
                
                # Set current version
                self.current_version = history_data.get('current_version')
                
                logger.info(f"Loaded {len(self.versions)} versions from history")
            else:
                logger.info("No version history found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load version history: {str(e)}")
            raise
    
    def _save_version_history(self) -> None:
        """Save version history to disk."""
        try:
            history_data = {
                'versions': [
                    {
                        'id': v.id,
                        'timestamp': v.timestamp,
                        'description': v.description,
                        'metadata': v.metadata,
                        'parent_id': v.parent_id
                    }
                    for v in self.versions.values()
                ],
                'current_version': self.current_version
            }
            
            with open(self.versions_path / "history.json", 'w') as f:
                json.dump(history_data, f, indent=2)
                
            logger.info("Saved version history")
            
        except Exception as e:
            logger.error(f"Failed to save version history: {str(e)}")
            raise
    
    def create_version(
        self,
        state: np.ndarray,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create new version of quantum state.
        
        Args:
            state: Quantum state to version
            description: Version description
            metadata: Optional metadata
            
        Returns:
            Version ID
        """
        try:
            # Generate version ID
            timestamp = datetime.datetime.now().strftime(VERSION_FORMAT)
            version_id = f"v_{timestamp}"
            
            # Create version object
            version = Version(
                id=version_id,
                timestamp=timestamp,
                description=description,
                metadata=metadata or {},
                parent_id=self.current_version
            )
            
            # Save state data
            state_path = self.versions_path / f"{version_id}.npy"
            np.save(state_path, state)
            
            # Update version history
            self.versions[version_id] = version
            self.current_version = version_id
            
            # Save history
            self._save_version_history()
            
            logger.info(f"Created version {version_id}: {description}")
            
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to create version: {str(e)}")
            raise
    
    def load_version(self, version_id: str) -> np.ndarray:
        """
        Load quantum state from version.
        
        Args:
            version_id: Version to load
            
        Returns:
            Quantum state
        """
        try:
            if version_id not in self.versions:
                raise ValueError(f"Unknown version: {version_id}")
            
            state_path = self.versions_path / f"{version_id}.npy"
            state = np.load(state_path)
            
            self.current_version = version_id
            logger.info(f"Loaded version {version_id}")
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to load version {version_id}: {str(e)}")
            raise
    
    def get_version_info(self, version_id: str) -> Version:
        """
        Get version information.
        
        Args:
            version_id: Version to query
            
        Returns:
            Version information
        """
        if version_id not in self.versions:
            raise ValueError(f"Unknown version: {version_id}")
        return self.versions[version_id]
    
    def list_versions(self) -> List[Version]:
        """
        List all versions.
        
        Returns:
            List of versions
        """
        return list(self.versions.values())
    
    def delete_version(self, version_id: str) -> None:
        """
        Delete version.
        
        Args:
            version_id: Version to delete
        """
        try:
            if version_id not in self.versions:
                raise ValueError(f"Unknown version: {version_id}")
            
            # Remove state file
            state_path = self.versions_path / f"{version_id}.npy"
            if state_path.exists():
                state_path.unlink()
            
            # Update version history
            del self.versions[version_id]
            if self.current_version == version_id:
                self.current_version = None
            
            # Save history
            self._save_version_history()
            
            logger.info(f"Deleted version {version_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete version {version_id}: {str(e)}")
            raise
    
    def cleanup(self) -> None:
        """Clean up version control system."""
        try:
            if self.versions_path.exists():
                shutil.rmtree(self.versions_path)
            logger.info("Cleaned up version control system")
            
        except Exception as e:
            logger.error(f"Failed to clean up version control: {str(e)}")
            raise