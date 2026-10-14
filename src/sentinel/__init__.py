"""SENTINEL v2: Two-Speed Memory Architecture (AERIS_64).

Parallel Form is the production ``forward()`` (depth passes over the sequence).
Distributed: ``torchrun`` + ``sentinel.training.distributed``.
"""

from .config import (
    AblationConfig,
    ModelConfig,
    SENTINELConfig,
    NeuroFieldConfig,
    SafetyConfig,
    load_config,
)
from .model import ModelOutput, SENTINEL, NeuroField
from .training import Trainer

__version__ = "2.5.1"
__all__ = [
    "SENTINEL",
    "NeuroField",
    "ModelOutput",
    "SENTINELConfig",
    "NeuroFieldConfig",
    "ModelConfig",
    "SafetyConfig",
    "AblationConfig",
    "load_config",
    "Trainer",
    "__version__",
]
