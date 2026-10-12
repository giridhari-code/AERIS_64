"""AERIS_64 — product shell with decoder-only Transformer backbone (default).

Legacy AESC/NeuroField modules remain importable under neurofield.modules.
"""

from .config import (
    AblationConfig,
    ModelConfig,
    NeuroFieldConfig,
    SafetyConfig,
    load_config,
)
from .model import ModelOutput, NeuroField
from .training import Trainer

__version__ = "3.0.0"
__all__ = [
    "NeuroField",
    "ModelOutput",
    "NeuroFieldConfig",
    "ModelConfig",
    "SafetyConfig",
    "AblationConfig",
    "load_config",
    "Trainer",
    "__version__",
]
