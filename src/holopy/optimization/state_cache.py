"""
Advanced caching system for holographic state computations.
"""
from typing import Dict, Tuple, Optional, Any
import numpy as np
import logging
from collections import OrderedDict
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class CacheMetrics:
    """Metrics for cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_bytes: int = 0
    avg_lookup_time: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class LRUStateCache:
    """LRU cache for quantum states with performance tracking."""
    
    def __init__(
        self,
        maxsize: int = 1000,
        maxbytes: Optional[int] = None
    ):
        self.maxsize = maxsize
        self.maxbytes = maxbytes
        self.cache: OrderedDict = OrderedDict()
        self.size_map: Dict[Tuple, int] = {}
        self.metrics = CacheMetrics()
        self.lookup_times: list = []
        
        logger.info(
            f"Initialized LRUStateCache with maxsize={maxsize}, "
            f"maxbytes={maxbytes if maxbytes else 'unlimited'}"
        )
    
    def get(
        self,
        key: Tuple[float, ...],
        default: Any = None
    ) -> Optional[np.ndarray]:
        """Retrieve state from cache with performance tracking."""
        start_time = time.perf_counter()
        
        try:
            value = self.cache.get(key, default)
            if value is not None:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.metrics.hits += 1
            else:
                self.metrics.misses += 1
                
            # Update timing metrics
            lookup_time = time.perf_counter() - start_time
            self.lookup_times.append(lookup_time)
            self.metrics.avg_lookup_time = np.mean(self.lookup_times[-100:])
            
            return value
            
        except Exception as e:
            logger.error(f"Cache lookup failed for key {key}: {str(e)}")
            return default
    
    def put(
        self,
        key: Tuple[float, ...],
        value: np.ndarray
    ) -> None:
        """Store state in cache with size management."""
        try:
            # Calculate size of new entry
            nbytes = value.nbytes
            
            # Check if we need to evict entries
            while (len(self.cache) >= self.maxsize or 
                   (self.maxbytes and self.metrics.total_bytes + nbytes > self.maxbytes)):
                if len(self.cache) == 0:
                    break
                    
                # Evict least recently used
                old_key, old_value = self.cache.popitem(last=False)
                old_size = self.size_map.pop(old_key)
                self.metrics.total_bytes -= old_size
                self.metrics.evictions += 1
                
            # Add new entry
            self.cache[key] = value
            self.size_map[key] = nbytes
            self.metrics.total_bytes += nbytes
            
        except Exception as e:
            logger.error(f"Cache insertion failed for key {key}: {str(e)}")
            raise
    
    def clear(self) -> None:
        """Clear cache and reset metrics."""
        self.cache.clear()
        self.size_map.clear()
        self.metrics = CacheMetrics()
        self.lookup_times.clear()
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current cache performance metrics."""
        return {
            'hit_rate': self.metrics.hit_rate,
            'total_entries': len(self.cache),
            'total_bytes': self.metrics.total_bytes,
            'evictions': self.metrics.evictions,
            'avg_lookup_time': self.metrics.avg_lookup_time
        } 