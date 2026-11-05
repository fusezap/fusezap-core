"""I81 D3: the FLUX.2 [klein] text encoder: Qwen3 (fusezap/serve/models/llama.py, encode()) -> prompt embeddings.

  enc = Qwen3TextEncoder.from_pretrained("black-forest-labs/FLUX.2-klein-4B", dtype=dtypes.float32, device="METAL")
  emb = enc.encode(["a red fox in the snow"])          # (B, 512, 3 * 2560) float32 numpy, diffusers' prompt_embeds; enc(...) too
  ids, mask = enc.tokenize(["a red fox in the snow"])  # (B, 512) int64 each

Reproduces diffusers 0.42 `Flux2KleinPipeline._get_qwen3_prompt_embeds`: each prompt goes through the chat template as
one user message with add_generation_prompt=True and enable_thinking=False, is tokenized without extra special tokens,
right-padded with the pad token to max_sequence_length (512) and truncated to it; the encoder runs with that 0/1
attention mask and HF's hidden_states[9], [18], [27] are stacked and laid out (B, T, 3 * D). Pad rows are kept as the
encoder computes them (the pipeline does not zero them). Only layers [0, 27) are loaded; no lm_head.
"""
from __future__ import annotations
import os
from typing import Sequence
import numpy as np
from fusezap.dtype import DType, dtypes

KLEIN_LAYERS = (9, 18, 27)
KLEIN_MAX_LEN = 512

def resolve_subfolders(name_or_path: str, subfolders: Sequence[str], allow_download: bool | None = None) -> str:
  """A diffusers-layout repo (model_index.json + component folders) as a local dir holding `subfolders`: a local path
  as is, else the HF cache snapshot (downloaded only if allow_download, default FZ_HF_DOWNLOAD=1)."""
  if os.path.isdir(name_or_path): return name_or_path
  from huggingface_hub import snapshot_download
  pats = ["model_index.json"] + [f"{s}/*" for s in subfolders]
  try: return snapshot_download(name_or_path, local_files_only=True, allow_patterns=pats)
  except Exception:
    if allow_download is None: allow_download = os.environ.get("FZ_HF_DOWNLOAD", "0") == "1"
    if not allow_download: raise
    return snapshot_download(name_or_path, allow_patterns=pats)

class Qwen3TextEncoder:
  def __init__(self, model, tokenizer, max_len: int = KLEIN_MAX_LEN, layers: Sequence[int] = KLEIN_LAYERS):
    """model: a fusezap Llama (Qwen3 config) loaded with at least max(layers) layers; tokenizer: fusezap Tokenizer."""
    self.model, self.tok, self.max_len, self.layers = model, tokenizer, int(max_len), tuple(layers)
    # eager, layer by layer (Llama.encode): bit-identical to the jit and faster for one call per prompt; the jit's capture
    # holds every intermediate (Qwen3-4B bf16 at 512 tokens: +16 GB peak, 9.5 GB kept), the eager pass peaks +0.9 GB
    if model is not None: model.jit_encode = False
    from fusezap.serve.tokenizer import hf_jinja_env, special_tokens
    assert tokenizer.template_src, "the tokenizer has no chat template"
    self._tpl, self._special = hf_jinja_env().from_string(tokenizer.template_src), special_tokens(tokenizer.config)
    assert tokenizer.pad_token_id is not None, "the tokenizer has no pad token"
    self.pad_id = int(tokenizer.pad_token_id)

  @staticmethod
  def from_pretrained(name_or_path: str, dtype: DType = dtypes.float32, device: str | None = None, max_len: int = KLEIN_MAX_LEN,
                      layers: Sequence[int] = KLEIN_LAYERS, text_encoder: str = "text_encoder", tokenizer: str = "tokenizer") -> "Qwen3TextEncoder":
    """A diffusers pipeline repo or dir (its text_encoder/ and tokenizer/ folders)."""
    from fusezap.serve.models.llama import Llama, LlamaConfig
    from fusezap.serve.tokenizer import Tokenizer
    from fusezap.serve.weights import load_safetensors, load_config
    root = resolve_subfolders(name_or_path, (text_encoder, tokenizer))
    te = os.path.join(root, text_encoder)
    cfg = LlamaConfig.from_hf(load_config(te))
    m = Llama(cfg, load_safetensors(te), dtype, device, max_len=max_len, lm_head=False, load_layers=max(layers))
    return Qwen3TextEncoder(m, Tokenizer(os.path.join(root, tokenizer)), max_len, layers)

  def prompt_text(self, prompt: str) -> str:
    return self._tpl.render(messages=[{"role": "user", "content": prompt}], tools=None, documents=None,
                            add_generation_prompt=True, enable_thinking=False, **self._special)

  def tokenize(self, prompts: str | Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """(B, max_len) ids and 0/1 mask: right padding with the pad token, truncation to max_len (HF padding='max_length')."""
    prompts = [prompts] if isinstance(prompts, str) else list(prompts)
    ids = np.full((len(prompts), self.max_len), self.pad_id, np.int64)
    mask = np.zeros((len(prompts), self.max_len), np.int64)
    for b, p in enumerate(prompts):
      t = self.tok.encode(self.prompt_text(p), add_special_tokens=False)[:self.max_len]
      ids[b, :len(t)], mask[b, :len(t)] = t, 1
    return ids, mask

  def encode_ids(self, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hs = self.model.encode(ids, mask, self.layers)                  # (B, n, T, D)
    B, n, T, D = hs.shape
    return np.ascontiguousarray(hs.transpose(0, 2, 1, 3).reshape(B, T, n * D))

  def encode(self, prompts: str | Sequence[str]) -> np.ndarray:
    """(B, max_len, len(layers) * D) float32: diffusers' prompt_embeds (values in the model dtype)."""
    return self.encode_ids(*self.tokenize(prompts))
  # the prompt-encoder seam of fusezap generate / Flux2KleinPipeline (I81 D6): enc(prompts) -> embeds, enc.release()
  __call__ = encode
  def release(self) -> None:
    """Drop the device weights (the caller loads the transformer next)."""
    self.model = None
    import gc; gc.collect()
