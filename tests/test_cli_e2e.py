"""B1-INT end-to-end gate: the installed `fusezap` console script (and `python -m fusezap`) serves Qwen2.5-0.5B-Instruct and
answers a streaming chat request from the official openai client. Marked slow (model load + warm-up per server process):
run with `uv run pytest -m slow tests/test_cli_e2e.py` or `FZ_SLOW=1 uv run pytest`."""
from __future__ import annotations
import os, shutil, socket, subprocess, sys, time
import pytest
from fusezap.device import Device

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
pytestmark = pytest.mark.slow

def _cached() -> bool:
  try:
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, local_files_only=True, allow_patterns=["config.json"]); return True
  except Exception: return False

def _free_port() -> int:
  with socket.socket() as s: s.bind(("127.0.0.1", 0)); return s.getsockname()[1]

def _wait(proc: subprocess.Popen, port: int, timeout: float = 600.0):
  import httpx
  t0 = time.time()
  while time.time() - t0 < timeout:
    if proc.poll() is not None: raise RuntimeError(f"server exited with {proc.returncode}:\n{proc.stdout.read()}")
    try:
      if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200: return
    except httpx.HTTPError: pass
    time.sleep(0.5)
  raise TimeoutError("server did not come up")

def _cmd(how: str, port: int, dev: str) -> list[str]:
  args = ["serve", MODEL, "--device", dev, "--port", str(port), "--max-model-len", "512", "--log-level", "warning"]
  if how == "script":
    exe = shutil.which("fusezap", path=os.path.dirname(sys.executable))
    if exe is None: pytest.skip("fusezap console script not installed (uv sync)")
    return [exe, *args]
  return [sys.executable, "-m", "fusezap", *args]

@pytest.mark.skipif(not _cached(), reason=f"{MODEL} not in the HF cache")
@pytest.mark.parametrize("how", ["script", "module"])
def test_serve_streams_to_openai_client(how):
  from openai import OpenAI
  dev = os.environ.get("FZ_DEVICE") or ("METAL" if Device.available("METAL") else "CPU")
  port = _free_port()
  proc = subprocess.Popen(_cmd(how, port, dev), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
  try:
    _wait(proc, port)
    c = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="unused")
    assert [m.id for m in c.models.list().data] == [MODEL]
    msgs = [{"role": "user", "content": "Name three primary colors."}]
    chunks = list(c.chat.completions.create(model=MODEL, messages=msgs, max_tokens=32, temperature=0, stream=True))
    deltas = [ch.choices[0].delta.content for ch in chunks if ch.choices and ch.choices[0].delta.content]
    assert len(deltas) > 3 and chunks[-1].choices[0].finish_reason in ("stop", "length")
    text = "".join(deltas)
    assert "red" in text.lower() and "blue" in text.lower()
    full = c.chat.completions.create(model=MODEL, messages=msgs, max_tokens=32, temperature=0)
    assert full.choices[0].message.content == text
  finally:
    proc.terminate()
    try: proc.wait(timeout=30)
    except subprocess.TimeoutExpired: proc.kill()
