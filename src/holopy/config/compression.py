from enum import Enum, auto

class CompressionMethod(Enum):
    """Available compression methods."""
    NONE = auto()
    ZLIB = auto()
    GZIP = auto()
    LZ4 = auto() 