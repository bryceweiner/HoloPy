"""
LRU cache implementation for quantum states.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
from collections import OrderedDict
import sys
from ..config.constants import (
    CACHE_SIZE,
    MAX_CACHE_ENTRIES,
    CACHE_CLEANUP_INTERVAL,
    CACHE_ALERT_THRESHOLD
)
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_used: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        """Convert metrics to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'bytes_used': self.bytes_used
        }

class LRUStateCache:
    """Least Recently Used (LRU) cache for quantum states."""
    
    def __init__(
        self,
        maxsize: int = MAX_CACHE_ENTRIES,
        maxbytes: int = CACHE_SIZE
    ):
        """
        Initialize cache.
        
        Args:
            maxsize: Maximum number of entries
            maxbytes: Maximum size in bytes
        """
        self.maxsize = maxsize
        self.maxbytes = maxbytes
        self._cache = {}
        self._bytes_used = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._access_count = {}
        self.metrics = CacheMetrics()
        
        logger.info(
            f"Initialized LRUStateCache with maxsize={maxsize}, "
            f"maxbytes={maxbytes}"
        )
    
    def get(self, key: Tuple) -> Optional[np.ndarray]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached array if present, None otherwise
        """
        try:
            value = self._cache.get(key)
            if value is not None:
                # Move to end (most recently used)
                self._access_count[key] = time.time()
                self._hits += 1
                return value
            
            self._misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed: {str(e)}")
            raise
    
    def put(self, key: Tuple, value: np.ndarray) -> bool:
        """
        Put item in cache.
        
        Args:
            key: Cache key
            value: Array to cache
            
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            # Calculate size of new entry
            new_size = sys.getsizeof(value.tobytes())
            
            # Check if single entry exceeds maxbytes
            if new_size > self.maxbytes:
                logger.warning(
                    f"Entry size {new_size} exceeds cache maxbytes {self.maxbytes}"
                )
                return False
            
            # Make space if needed
            while (
                self._bytes_used + new_size > self.maxbytes or
                len(self._cache) >= self.maxsize
            ):
                if len(self._cache) == 0:
                    break
                    
                # Remove least recently used
                lru_key = min(self._access_count.items(), key=lambda x: x[1])[0]
                lru_value = self._cache[lru_key]
                self._bytes_used -= sys.getsizeof(lru_value.tobytes())
                del self._cache[lru_key]
                del self._access_count[lru_key]
                self.metrics.evictions += 1
            
            # Remove existing entry if present
            if key in self._cache:
                old_value = self._cache[key]
                self._bytes_used -= sys.getsizeof(old_value.tobytes())
                del self._cache[key]
            
            # Add new entry
            self._cache[key] = value
            self._access_count[key] = time.time()
            self._bytes_used += new_size
            return True
            
        except Exception as e:
            logger.error(f"Cache put failed: {str(e)}")
            return False
    
    def clear(self) -> None:
        """Clear cache."""
        try:
            self._cache.clear()
            self._bytes_used = 0
            logger.info("Cache cleared")
            
        except Exception as e:
            logger.error(f"Cache clear failed: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict[str, int]:
        """
        Get cache metrics.
        
        Returns:
            Dictionary of metrics
        """
        try:
            total_requests = self._hits + self._misses
            hit_rate = (
                self._hits / total_requests if total_requests > 0 else 0
            )
            
            return {
                'size': len(self._cache),
                'maxsize': self.maxsize,
                'total_bytes': self._bytes_used,
                'maxbytes': self.maxbytes,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'bytes_used_pct': self._bytes_used / self.maxbytes
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}")
            raise
    
    def cleanup(self) -> None:
        """Perform cache cleanup."""
        try:
            # Remove least recently used entries until within limits
            while (
                self._bytes_used > self.maxbytes * CACHE_ALERT_THRESHOLD or
                len(self._cache) > self.maxsize * CACHE_ALERT_THRESHOLD
            ):
                if len(self._cache) == 0:
                    break
                old_value = self._cache  # Get the old value first    
                lru_key = min(self._access_count.items(), key=lambda x: x[1])[0]
                del self._cache[lru_key]
                del self._access_count[lru_key]
                self._bytes_used -= sys.getsizeof(old_value.tobytes())
                self._evictions += 1
            
            logger.info(
                f"Cache cleanup complete: {self._bytes_used}/{self.maxbytes} bytes, "
                f"{len(self._cache)}/{self.maxsize} entries"
            )
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {str(e)}")
            raise 

    def test_cache_eviction(self, setup_cache):
        """Test LRU eviction policy."""
        cache = setup_cache

        # Fill cache
        for i in range(15):  # More than maxsize
            key = (float(i),)
            value = np.random.random(100)
            cache.put(key, value)

        # Verify size constraint using _cache internal attribute
        assert len(cache._cache) <= cache.maxsize

    def _update_metrics(self, key: Tuple, value: np.ndarray) -> None:
        """Update cache metrics."""
        if key in self._cache:
            old_value = self._cache[key]  # Get the old value first
            self.metrics.update_count += 1
            self.metrics.update_size += value.nbytes - old_value.nbytes
        else:
            self.metrics.insert_count += 1
            self.metrics.insert_size += value.nbytes