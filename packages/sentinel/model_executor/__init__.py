"""Model executor — DeepInfer-inspired inference execution layer for SENTINEL.

Layout (mirrors deepinfer/model_executor/ roles, PyTorch-only):
  model_loader   load weights + tokenizer from checkpoint folder
  model_runner   prefill / decode steps over SENTINEL.forward
  sampling       temperature / top_p / top_k / seed
  output         runner step result types

Serve (serving/server.py) owns HTTP + Sequence lifecycle; this package owns
device execution only.
"""

from sentinel.model_executor.model_loader import LoadedModel, load_model
from sentinel.model_executor.model_runner import ModelRunner, RunnerConfig
from sentinel.model_executor.output import ModelStepOutput
from sentinel.model_executor.sampling import SamplingParams, sample_from_logits

__all__ = [
    "LoadedModel",
    "load_model",
    "ModelRunner",
    "RunnerConfig",
    "ModelStepOutput",
    "SamplingParams",
    "sample_from_logits",
]
