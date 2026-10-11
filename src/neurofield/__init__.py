"""NeuroField / AESC v2: Two-Speed Memory Architecture (AERIS_64).

Sequential ``forward()`` is the production default.
Distributed: ``torchrun`` + ``neurofield.training.distributed``.
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

__version__ = "2.4.0"
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
