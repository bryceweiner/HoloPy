"""Unit tests for state caching system."""
import pytest
import numpy as np
from holopy.optimization.state_cache import LRUStateCache
from holopy.core.propagator import CachedPropagator

class TestStateCache:
    @pytest.fixture
    def setup_cache(self):
        """Setup test cache system."""
        return LRUStateCache(maxsize=10, maxbytes=1024*1024)
    
    def test_basic_caching(self, setup_cache):
        """Test basic cache operations."""
        cache = setup_cache
        
        # Create test data
        key = (0.1, 0.2)
        value = np.random.random(100)
        
        # Test insertion
        cache.put(key, value)
        retrieved = cache.get(key)
        assert np.array_equal(value, retrieved)
        
        # Test cache hit tracking
        _ = cache.get(key)
        metrics = cache.get_metrics()
        assert metrics['hit_rate'] == 2/2  # 2 hits, 0 misses
        
    def test_cache_eviction(self, setup_cache):
        """Test LRU eviction policy."""
        cache = setup_cache
        
        # Fill cache
        for i in range(15):  # More than maxsize
            key = (float(i),)
            value = np.random.random(100)
            cache.put(key, value)
        
        # Verify size constraint
        assert len(cache.cache) <= cache.maxsize
        
        # Verify LRU behavior
        metrics = cache.get_metrics()
        assert metrics['evictions'] >= 5  # Should have evicted at least 5 entries
        
    def test_size_constraints(self, setup_cache):
        """Test byte size constraints."""
        cache = setup_cache
        
        # Create large array
        large_array = np.random.random(1000000)  # Should exceed maxbytes
        
        # Attempt to cache
        cache.put((0.1,), large_array)
        
        # Verify size constraint
        metrics = cache.get_metrics()
        assert metrics['total_bytes'] <= cache.maxbytes

class TestCachedPropagator:
    @pytest.fixture
    def setup_propagator(self):
        """Setup test propagator."""
        return CachedPropagator(
            spatial_points=128,
            spatial_extent=10.0,
            cache_size=100
        )
    
    def test_propagator_caching(self, setup_propagator):
        """Test propagator caching behavior."""
        propagator = setup_propagator
        
        # First call - should compute
        prop1 = propagator.get_propagator(0.1, 1e-29)
        
        # Second call - should hit cache
        prop2 = propagator.get_propagator(0.1, 1e-29)
        
        # Verify cache hit
        metrics = propagator.cache.get_metrics()
        assert metrics['hit_rate'] > 0
        
        # Verify same result
        assert np.array_equal(prop1, prop2)
    
    def test_propagation_with_cache(self, setup_propagator):
        """Test full propagation with caching."""
        propagator = setup_propagator
        
        # Create test wavefunction
        x = np.linspace(-5, 5, 128)
        psi = np.exp(-x**2/2)  # Gaussian
        
        # Evolve multiple times
        evolved1 = propagator.propagate(psi, 0.1)
        evolved2 = propagator.propagate(psi, 0.1)
        
        # Verify results match
        assert np.allclose(evolved1, evolved2)
        
        # Verify cache was used
        metrics = propagator.cache.get_metrics()
        assert metrics['hit_rate'] > 0 