"""
Physical and computational constants for holographic quantum systems.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

# Fundamental Physical Constants
PLANCK_CONSTANT = 6.62607015e-34  # Planck constant (J⋅s)
REDUCED_PLANCK = PLANCK_CONSTANT / (2 * np.pi)  # Reduced Planck constant (J⋅s)
SPEED_OF_LIGHT = 2.99792458e8  # Speed of light (m/s)
BOLTZMANN_CONSTANT = 1.380649e-23  # Boltzmann constant (J/K)
GRAVITATIONAL_CONSTANT = 6.67430e-11  # Gravitational constant (m³/kg⋅s²)
VACUUM_PERMITTIVITY = 8.8541878128e-12  # Vacuum permittivity (F/m)

# Derived Physical Constants
PLANCK_LENGTH = np.sqrt(REDUCED_PLANCK * GRAVITATIONAL_CONSTANT / SPEED_OF_LIGHT**3)  # Planck length (m)
PLANCK_TIME = PLANCK_LENGTH / SPEED_OF_LIGHT  # Planck time (s)
PLANCK_MASS = np.sqrt(REDUCED_PLANCK * SPEED_OF_LIGHT / GRAVITATIONAL_CONSTANT)  # Planck mass (kg)
PLANCK_TEMPERATURE = np.sqrt(REDUCED_PLANCK * SPEED_OF_LIGHT**5 / (GRAVITATIONAL_CONSTANT * BOLTZMANN_CONSTANT**2))  # Planck temperature (K)

# Holographic Parameters
COUPLING_CONSTANT = 0.1  # Holographic coupling strength
INFORMATION_GENERATION_RATE = 1e-3  # Information generation rate
CRITICAL_THRESHOLD = 0.1  # Critical threshold for error correction
SPATIAL_EXTENT = 1.0  # Default spatial extent

# Dimensional Parameters
E8_DIMENSION = 248  # Dimension of E8 lattice
TOTAL_DIMENSION = 496  # Total dimension of the system (2 * E8_DIMENSION)
BOUNDARY_DIMENSION = E8_DIMENSION - 1  # Dimension of the boundary
BULK_DIMENSION = E8_DIMENSION + 1  # Dimension of the bulk

# Evolution Parameters
TARGET_EVOLUTION_TIME = 1e-12  # Target evolution time (s)
EVOLUTION_STEPS = 1000  # Number of evolution steps
TIME_STEP = TARGET_EVOLUTION_TIME / EVOLUTION_STEPS  # Time step size
EVOLUTION_TOLERANCE = 1e-6  # Evolution convergence tolerance

# Computational Parameters
DEFAULT_GRID_POINTS = 128  # Default number of grid points
DEFAULT_TIME_STEP = 1e-15  # Default time step (s)
DEFAULT_TOLERANCE = 1e-10  # Default numerical tolerance
MAX_ITERATIONS = 1000  # Maximum iterations for optimization

# System Parameters
DEFAULT_TEMPERATURE = 300.0  # Default temperature (K)
DEFAULT_PRESSURE = 101325.0  # Default pressure (Pa)
AMBIENT_NOISE = 1e-6  # Ambient noise level

# Hierarchy Parameters
HIERARCHY_LEVELS = 7  # Number of levels in information hierarchy
LEVEL_COUPLING = 0.5  # Coupling strength between hierarchy levels
INFORMATION_FLOW_RATE = 1e-3  # Rate of information flow between levels

# Version Control
VERSION_FORMAT = "%Y%m%d_%H%M%S"  # Format for version timestamps
MAX_VERSIONS = 100  # Maximum number of versions to keep
VERSION_METADATA_SCHEMA: Dict[str, Any] = {
    "author": str,
    "description": str,
    "timestamp": str,
    "parameters": dict,
    "metrics": dict
}

# File System
DEFAULT_DATA_DIR = Path.home() / ".holopy"
CACHE_DIR = DEFAULT_DATA_DIR / "cache"
STATES_DIR = DEFAULT_DATA_DIR / "states"
LOGS_DIR = DEFAULT_DATA_DIR / "logs"

# Logging Configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE_PATH = "logs/holopy.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5
LOG_TO_CONSOLE = True
LOG_TO_FILE = True

# Logging Levels
DEBUG_LOG_LEVEL = logging.DEBUG
INFO_LOG_LEVEL = logging.INFO
WARNING_LOG_LEVEL = logging.WARNING
ERROR_LOG_LEVEL = logging.ERROR
CRITICAL_LOG_LEVEL = logging.CRITICAL

# Log Categories
QUANTUM_LOGGER = "holopy.quantum"
OPTIMIZATION_LOGGER = "holopy.optimization"
CORE_LOGGER = "holopy.core"
UTILS_LOGGER = "holopy.utils"
CONFIG_LOGGER = "holopy.config"

# Log File Paths
QUANTUM_LOG_FILE = "logs/quantum.log"
OPTIMIZATION_LOG_FILE = "logs/optimization.log"
CORE_LOG_FILE = "logs/core.log"
UTILS_LOG_FILE = "logs/utils.log"
CONFIG_LOG_FILE = "logs/config.log"

# Error Correction
MAX_CODE_DISTANCE = 50
MIN_CODE_DISTANCE = 3
DEFAULT_CODE_DISTANCE = 7
SYNDROME_THRESHOLD = 0.1

# Quantum Circuit
MAX_QUBITS = 64
MAX_GATES_PER_LAYER = 1000
MAX_CIRCUIT_DEPTH = 1000

# Optimization
LEARNING_RATE = 0.01
MOMENTUM = 0.9
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Visualization
PLOT_DPI = 300
FIGURE_SIZE = (10, 6)
COLOR_MAP = 'viridis'
FONT_SIZE = 12

# Performance
CACHE_SIZE = 1024 * 1024  # Default cache size in bytes (1 MB)
MAX_CACHE_ENTRIES = 10000  # Maximum number of cache entries
CACHE_CLEANUP_INTERVAL = 300  # Cache cleanup interval in seconds
CACHE_ALERT_THRESHOLD = 0.8  # Cache usage alert threshold (80%)
CACHE_EVICTION_BATCH = 100  # Number of entries to evict at once
CACHE_PRELOAD_SIZE = 1000  # Number of entries to preload
CACHE_WARMUP_TIME = 60  # Cache warmup time in seconds
CACHE_STATS_WINDOW = 1000  # Window size for cache statistics

# Cache Performance Settings
CACHE_HIT_TARGET = 0.9  # Target cache hit rate
CACHE_MISS_THRESHOLD = 0.2  # Maximum acceptable miss rate
CACHE_LATENCY_TARGET = 0.001  # Target cache latency in seconds
CACHE_MEMORY_LIMIT = 0.25  # Maximum memory usage as fraction of system RAM
CACHE_CPU_LIMIT = 0.1  # Maximum CPU usage for cache operations
CACHE_IO_BATCH_SIZE = 1024 * 1024  # I/O batch size for cache operations

# Cache Persistence
CACHE_PERSISTENCE_ENABLED = True  # Enable cache persistence
CACHE_BACKUP_INTERVAL = 3600  # Cache backup interval in seconds
CACHE_BACKUP_COUNT = 3  # Number of cache backups to maintain
CACHE_RECOVERY_TIMEOUT = 30  # Cache recovery timeout in seconds
CACHE_SYNC_INTERVAL = 60  # Cache synchronization interval

# Cache Security
CACHE_ENCRYPTION_ENABLED = False  # Enable cache encryption
CACHE_HASH_ALGORITHM = 'sha256'  # Hash algorithm for cache keys
CACHE_KEY_DERIVATION = 'pbkdf2'  # Key derivation function
CACHE_SALT_SIZE = 16  # Salt size in bytes
CACHE_MAC_SIZE = 32  # MAC size in bytes

# Cache Monitoring
CACHE_MONITOR_INTERVAL = 60  # Monitoring interval in seconds
CACHE_LOG_LEVEL = 'INFO'  # Cache logging level
CACHE_METRIC_WINDOW = 3600  # Metric collection window
CACHE_ALERT_COOLDOWN = 300  # Alert cooldown period
CACHE_HEALTH_CHECK_INTERVAL = 60  # Health check interval

# Cache Distribution
CACHE_SHARDS = 16  # Number of cache shards
CACHE_REPLICATION_FACTOR = 2  # Cache replication factor
CACHE_CONSISTENCY_LEVEL = 'QUORUM'  # Cache consistency level
CACHE_PARTITION_SIZE = 1024 * 1024  # Size of each cache partition
CACHE_ROUTING_ALGORITHM = 'consistent-hashing'  # Cache routing algorithm

# Cache Optimization
CACHE_PREFETCH_ENABLED = True  # Enable cache prefetching
CACHE_COMPRESSION_ENABLED = True  # Enable cache compression
CACHE_DEDUP_ENABLED = True  # Enable cache deduplication
CACHE_ADAPTIVE_SIZING = True  # Enable adaptive cache sizing
CACHE_LEARNING_RATE = 0.1  # Learning rate for adaptive sizing

# Threading and Concurrency
MIN_THREAD_POOL_SIZE = 2  # Minimum thread pool size
MAX_THREAD_POOL_SIZE = 16  # Maximum thread pool size
THREAD_TIMEOUT = 30  # Thread timeout in seconds
THREAD_KEEPALIVE = 60  # Thread keepalive time in seconds
MAX_QUEUE_SIZE = 1000  # Maximum task queue size
WORKER_BATCH_SIZE = 50  # Batch size for worker threads

# Resource Management
CPU_AFFINITY_MASK = 0xFFFF  # CPU affinity mask for process
MEMORY_LIMIT = 8 * 1024 * 1024 * 1024  # Memory limit (8 GB)
DISK_BUFFER_SIZE = 8192  # Disk I/O buffer size
MAX_FILE_DESCRIPTORS = 1024  # Maximum number of file descriptors
IO_TIMEOUT = 5.0  # I/O operation timeout in seconds
RESOURCE_CHECK_INTERVAL = 10  # Resource check interval in seconds

# Profiling and Monitoring
PROFILE_SAMPLE_INTERVAL = 0.001  # Profiling sample interval in seconds
TRACE_LOG_SIZE = 1000  # Maximum size of trace log
METRICS_RETENTION_PERIOD = 86400  # Metrics retention period in seconds
MONITORING_ENABLED = True  # Enable performance monitoring
DEBUG_PERFORMANCE = False  # Enable detailed performance debugging

# Optimization Settings
JIT_THRESHOLD = 100  # Number of calls before JIT compilation
VECTORIZATION_THRESHOLD = 1000  # Minimum size for vectorization
PARALLEL_THRESHOLD = 10000  # Minimum size for parallel execution
OPTIMIZATION_LEVEL = 3  # Optimization level (0-3)
AUTO_TUNE_INTERVAL = 3600  # Auto-tuning interval in seconds

# Network Performance
NETWORK_BUFFER_SIZE = 65536  # Network buffer size
MAX_CONNECTIONS = 1000  # Maximum concurrent connections
CONNECTION_TIMEOUT = 30  # Connection timeout in seconds
KEEPALIVE_INTERVAL = 60  # Keepalive interval in seconds
BANDWIDTH_LIMIT = 1000000  # Bandwidth limit in bytes per second
PACKET_SIZE = 1472  # Maximum packet size

# Database Performance
DB_CONNECTION_POOL_SIZE = 10  # Database connection pool size
DB_TIMEOUT = 5  # Database operation timeout in seconds
DB_MAX_RETRIES = 3  # Maximum database retry attempts
DB_BATCH_SIZE = 1000  # Database batch operation size
DB_CACHE_SIZE = 1000  # Database query cache size

# Security
HASH_ALGORITHM = 'sha256'
ENCRYPTION_ALGORITHM = 'AES-256-GCM'
KEY_SIZE = 32  # bytes

# Network
DEFAULT_PORT = 8080
TIMEOUT = 30  # seconds
MAX_RETRIES = 3
BUFFER_SIZE = 8192  # bytes

# Documentation
DOC_URL = "https://holopy.readthedocs.io"
REPO_URL = "https://github.com/username/holopy"
LICENSE = "MIT"
VERSION = "0.1.0"

# Compression Settings
COMPRESSION_LEVEL = 6  # Default compression level (0-9)
DEFAULT_COMPRESSION_METHOD = "zlib"  # Default compression method
COMPRESSION_THRESHOLD = 1024  # Minimum size for compression (bytes)
COMPRESSION_CHUNK_SIZE = 65536  # Chunk size for streaming compression
MAX_COMPRESSION_THREADS = 4  # Maximum threads for parallel compression

# Compression Methods Configuration
COMPRESSION_METHODS = {
    "zlib": {
        "min_level": 0,
        "max_level": 9,
        "default_level": 6,
        "compression_type": "lossless"
    },
    "lzma": {
        "min_level": 0,
        "max_level": 9,
        "default_level": 6,
        "compression_type": "lossless"
    },
    "bzip2": {
        "min_level": 1,
        "max_level": 9,
        "default_level": 6,
        "compression_type": "lossless"
    },
    "wavelet": {
        "min_level": 1,
        "max_level": 5,
        "default_level": 3,
        "compression_type": "lossy"
    }
}

# Compression Quality Settings
LOSSY_COMPRESSION_TOLERANCE = 1e-6  # Maximum allowed error for lossy compression
COMPRESSION_QUALITY_LEVELS = {
    "high": {
        "level": 9,
        "tolerance": 1e-8
    },
    "medium": {
        "level": 6,
        "tolerance": 1e-6
    },
    "low": {
        "level": 3,
        "tolerance": 1e-4
    }
}

# Compression Performance Settings
COMPRESSION_CACHE_SIZE = 1024 * 1024  # Size of compression cache (bytes)
COMPRESSION_BUFFER_SIZE = 8192  # Size of compression buffer (bytes)
MIN_COMPRESSION_RATIO = 1.2  # Minimum compression ratio to store compressed
MAX_COMPRESSION_TIME = 5.0  # Maximum time for compression (seconds)
COMPRESSION_TIMEOUT = 30.0  # Timeout for compression operations

# Compression Monitoring
COMPRESSION_STATS_WINDOW = 1000  # Window size for compression statistics
COMPRESSION_ALERT_THRESHOLD = 0.8  # Alert threshold for compression issues
COMPRESSION_LOG_INTERVAL = 3600  # Interval for compression log rotation (seconds)

# Performance Targets
TARGET_CACHE_HIT_RATE = 0.95  # Target cache hit rate (95%)
TARGET_CACHE_MISS_RATE = 0.05  # Target cache miss rate (5%)
TARGET_CACHE_EVICTION_RATE = 0.01  # Target cache eviction rate (1%)
TARGET_CACHE_UTILIZATION = 0.80  # Target cache utilization (80%)
TARGET_CACHE_RESPONSE_TIME = 0.001  # Target cache response time (1ms)
TARGET_MEMORY_UTILIZATION = 0.75  # Target memory utilization (75%)
TARGET_CPU_UTILIZATION = 0.50  # Target CPU utilization (50%)
TARGET_DISK_UTILIZATION = 0.70  # Target disk utilization (70%)
TARGET_NETWORK_UTILIZATION = 0.60  # Target network utilization (60%)
TARGET_THROUGHPUT = 10000  # Target operations per second
TARGET_LATENCY = 0.010  # Target latency in seconds (10ms)
TARGET_ERROR_RATE = 0.001  # Target error rate (0.1%)
TARGET_AVAILABILITY = 0.999  # Target availability (99.9%)
TARGET_CONSISTENCY_LEVEL = 0.95  # Target consistency level (95%)

# Performance Thresholds
CRITICAL_CACHE_HIT_RATE = 0.80  # Critical cache hit rate threshold
CRITICAL_MEMORY_USAGE = 0.90  # Critical memory usage threshold
CRITICAL_CPU_USAGE = 0.85  # Critical CPU usage threshold
CRITICAL_DISK_USAGE = 0.90  # Critical disk usage threshold
CRITICAL_NETWORK_USAGE = 0.85  # Critical network usage threshold
CRITICAL_ERROR_RATE = 0.05  # Critical error rate threshold
CRITICAL_LATENCY = 0.100  # Critical latency threshold (100ms)
CRITICAL_QUEUE_LENGTH = 1000  # Critical queue length threshold

# Performance Monitoring
PERFORMANCE_SAMPLE_INTERVAL = 1.0  # Performance sampling interval (seconds)
PERFORMANCE_WINDOW_SIZE = 3600  # Performance monitoring window (1 hour)
PERFORMANCE_METRIC_COUNT = 100  # Number of metrics to track
PERFORMANCE_LOG_INTERVAL = 300  # Performance logging interval (5 minutes)
PERFORMANCE_ALERT_COOLDOWN = 600  # Alert cooldown period (10 minutes)
PERFORMANCE_REPORT_INTERVAL = 3600  # Performance report interval (1 hour)

# Performance Optimization
OPTIMIZATION_INTERVAL = 300  # Optimization interval (5 minutes)
OPTIMIZATION_WINDOW = 3600  # Optimization window (1 hour)
OPTIMIZATION_THRESHOLD = 0.10  # Minimum improvement threshold
OPTIMIZATION_MAX_ATTEMPTS = 5  # Maximum optimization attempts
OPTIMIZATION_COOLDOWN = 600  # Optimization cooldown period
OPTIMIZATION_BATCH_SIZE = 1000  # Optimization batch size

# Performance Scaling
AUTO_SCALE_ENABLED = True  # Enable auto-scaling
SCALE_UP_THRESHOLD = 0.85  # Scale up threshold (85% utilization)
SCALE_DOWN_THRESHOLD = 0.25  # Scale down threshold (25% utilization)
SCALE_FACTOR = 1.5  # Scale factor for resource adjustment
MIN_SCALE_UNITS = 1  # Minimum scale units
MAX_SCALE_UNITS = 10  # Maximum scale units
SCALE_COOLDOWN = 300  # Scaling cooldown period

# Performance Debugging
DEBUG_PERFORMANCE = False  # Enable performance debugging
DEBUG_SAMPLE_RATE = 0.01  # Debug sampling rate
DEBUG_LOG_LEVEL = 'INFO'  # Debug log level
DEBUG_METRIC_COUNT = 1000  # Number of debug metrics
DEBUG_RETENTION_PERIOD = 86400  # Debug data retention period
DEBUG_EXPORT_INTERVAL = 3600  # Debug export interval