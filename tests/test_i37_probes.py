"""scripts/i37_failure_probes.py (I37) against a fake OpenAI-style server: the probe logic, not a real engine."""
import importlib.util, json, os, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_spec = importlib.util.spec_from_file_location("i37", os.path.join(os.path.dirname(__file__), "..", "scripts", "i37_failure_probes.py"))
P = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P)

class _Fake(BaseHTTPRequestHandler):
  L, bad5xx = 64, False
  def log_message(self, *a): pass
  def _send(self, code: int, obj) -> None:
    b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
  def do_GET(self):
    if self.path == "/v1/models": self._send(200, {"data": [{"id": "m", "max_model_len": self.L}]})
    else: self._send(200, {"status": "ok"})
  def do_POST(self):
    raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
    try: body = json.loads(raw)
    except ValueError: return self._send(500 if self.bad5xx else 400, {"error": {"message": "bad json"}})
    if body.get("model") != "m": return self._send(404, {"error": {"message": "no model"}})
    if body.get("max_tokens", 1) < 1: return self._send(400, {"error": {"message": "max_tokens"}})
    p = body["prompt"]
    if isinstance(p, list) and len(p) > self.L: return self._send(400, {"error": {"code": "context_length_exceeded"}})
    self._send(200, {"choices": [{"text": str(sum(p) if isinstance(p, list) else len(p))}]})

def _serve(bad5xx=False):
  h = type("H", (_Fake,), {"bad5xx": bad5xx})
  s = ThreadingHTTPServer(("127.0.0.1", 0), h)
  threading.Thread(target=s.serve_forever, daemon=True).start()
  return s

def _args(url, **kw):
  from types import SimpleNamespace
  return SimpleNamespace(base_url=url, model="m", pid=0, pressure_requests=6, pressure_prompt=16, pressure_tokens=4, pressure_check=3, crash_timeout=1.0, **kw)

def test_probes_pass_on_a_well_behaved_server():
  s = _serve()
  try:
    a = _args(f"http://127.0.0.1:{s.server_address[1]}")
    assert P.probe_bad(a)["status"] == "pass"
    r = P.probe_overlength(a)
    assert r["status"] == "pass" and r["http"] == 400 and r["code"] == "context_length_exceeded" and r["prompt_tokens"] == 64 + 64
    r = P.probe_pressure(a)
    assert r["status"] == "pass" and r["n_errors"] == 0 and r["alone_equal"] == "3/3"
    assert P.probe_crash(a, None)["status"] == "skip"   # no --pid
  finally: s.shutdown()

def test_bad_probe_fails_on_a_5xx():
  s = _serve(bad5xx=True)
  try:
    r = P.probe_bad(_args(f"http://127.0.0.1:{s.server_address[1]}"))
    assert r["status"] == "fail" and r["codes"]["not JSON"] == 500
  finally: s.shutdown()

def test_dropped_connection_is_status_0():
  st, why = P.http("http://127.0.0.1:9", "/health", None, 2)
  assert st == 0 and why
