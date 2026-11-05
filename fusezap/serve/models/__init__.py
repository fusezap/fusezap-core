"""Model registry: HF `model_type` -> implementation. Every supported architecture is a Llama-style decoder."""
from __future__ import annotations
from fusezap.dtype import DType, dtypes
from fusezap.serve.models.llama import Llama, LlamaConfig, KVCache, ContiguousKVCache  # noqa: F401

ARCHITECTURES: dict[str, type] = {"llama": Llama, "qwen2": Llama, "mistral": Llama, "qwen3": Llama}

def load_model(name_or_path: str, dtype: DType = dtypes.float32, device: str | None = None, **kw) -> Llama:
  from fusezap.serve.weights import resolve_model, load_config
  mt = load_config(resolve_model(name_or_path)).get("model_type", "llama")
  if mt not in ARCHITECTURES: raise NotImplementedError(f"model_type {mt!r} is not supported (have {sorted(ARCHITECTURES)})")
  return ARCHITECTURES[mt].from_pretrained(name_or_path, dtype=dtype, device=device, **kw)
