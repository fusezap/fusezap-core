"""Talk to a running FuseZap server with the official OpenAI client.

  uv run python -m fusezap serve Qwen/Qwen2.5-0.5B-Instruct --device METAL --port 8000     # in one terminal
  uv run python examples/openai_client.py [--base-url http://127.0.0.1:8000/v1] [--model Qwen/Qwen2.5-0.5B-Instruct]
"""
import argparse
from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--api-key", default="unused")
a = ap.parse_args()
client = OpenAI(base_url=a.base_url, api_key=a.api_key)

print("models:", [m.id for m in client.models.list().data])
msgs = [{"role": "user", "content": "Name three primary colors."}]
r = client.chat.completions.create(model=a.model, messages=msgs, max_tokens=64, temperature=0)
print("chat:", r.choices[0].message.content, "|", r.usage)
print("stream: ", end="", flush=True)
for ch in client.chat.completions.create(model=a.model, messages=msgs, max_tokens=64, temperature=0, stream=True):
  print(ch.choices[0].delta.content or "", end="", flush=True)
print()
print("completion:", client.completions.create(model=a.model, prompt="The capital of France is", max_tokens=8, temperature=0).choices[0].text)
