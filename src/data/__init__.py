"""Data collection and processing modules"""

from .affine_data import AffineDataCollector
from .agentgym_data import AgentGymDataCollector

# Make dataset imports optional (requires torch)
try:
    from .dataset import TrainingDataset, RLDataset
    __all__ = [
        "AffineDataCollector",
        "AgentGymDataCollector",
        "TrainingDataset",
        "RLDataset",
    ]
except ImportError:
    # torch not available - dataset classes won't be available
    # but data collection classes still work
    __all__ = [
        "AffineDataCollector",
        "AgentGymDataCollector",
    ]
