"""The store contract (STORE_API) and the record schema (RECORD_VERSION).

A record is one run, as plain JSON-able data:

  {"record_version": 1,
   "run": {key, run_id, name, label, started_utc, finished_utc, tool_version, mode, engine, engine_version, model,
           hardware, dtype, workload, load_points, group_key, verdict, decision, baseline, anchor, source, extra},
   "gates":   [{gate_id, check, kind, against, status, metric, comparator, threshold, value, baseline_value, ci_lo, ci_hi}],
   "metrics": [{name, profile, concurrency, value}]}

run.key is the sha256 (64 hex characters) of the run's report file: the identity of a run in every store. Strings
may be None except key and group_key; run.extra is a JSON object for fields a later record version promotes to
columns. gates: one per gate of the run (threshold as text: it can be a number or a word such as "pass"). metrics:
one per measured number, `profile` "" and `concurrency` 0 when the metric is not per profile / load point (e.g.
top1_agreement), so (key, name, profile, concurrency) is unique.

A store (STORE_API = 1) implements:

  migrate() -> int             create or upgrade its tables to the current schema; returns the schema version
  put_run(record) -> bool      store one record; False (and no change) when a run with that key is already stored
  query(group_key=None, model=None, since=None, limit=None) -> list[record]
                               records, oldest first (finished_utc, then key); since: finished_utc >= since (ISO 8601
                               text); limit: the newest `limit` of them, still oldest first
  keys() -> set[str]           the keys stored
  close()                      release the connection (stores are also context managers)

and sets `api = STORE_API` and `scheme` (the URL scheme it serves). A factory `open(url) -> Store` makes one from its
URL; open_store() checks the api, then calls migrate().
"""
from __future__ import annotations
import json, math, re
from typing import Any

STORE_API = 1
RECORD_VERSION = 1

RUN_FIELDS = ("key", "run_id", "name", "label", "started_utc", "finished_utc", "tool_version", "mode", "engine", "engine_version", "model",
              "hardware", "dtype", "workload", "load_points", "group_key", "verdict", "decision", "baseline", "anchor", "source", "extra")
GATE_FIELDS = ("gate_id", "check", "kind", "against", "status", "metric", "comparator", "threshold", "value", "baseline_value", "ci_lo", "ci_hi")
GATE_NUMBERS = ("value", "baseline_value", "ci_lo", "ci_hi")
METRIC_FIELDS = ("name", "profile", "concurrency", "value")
_KEY = re.compile(r"^[0-9a-f]{64}$")

class StoreError(Exception):
  """A store could not do what was asked (a bad URL, a schema newer than this code, a record that does not fit)."""

class MissingDriver(StoreError):
  """The database driver of a backend is not installed; the message says which extra installs it."""

class Store:
  """Base class of a store (optional: anything with these methods and `api` works). Context manager: closes on exit."""
  api: int = STORE_API
  scheme: str = ""
  def migrate(self) -> int: raise NotImplementedError
  def put_run(self, record: dict) -> bool: raise NotImplementedError
  def query(self, group_key: str | None = None, model: str | None = None, since: str | None = None, limit: int | None = None) -> list[dict]:
    raise NotImplementedError
  def keys(self) -> set[str]: raise NotImplementedError
  def close(self) -> None: pass
  def __enter__(self): return self
  def __exit__(self, *exc) -> None: self.close()

def _num(x: Any) -> float | None:
  if x is None or isinstance(x, bool): return None
  if isinstance(x, (int, float)):
    return None if isinstance(x, float) and (math.isnan(x) or math.isinf(x)) else float(x)
  return None

def _str(x: Any) -> str | None:
  return None if x is None else str(x)

def normalize_record(rec: dict) -> dict:
  """The record in canonical form (every field present, numbers as float or None, text as str or None, threshold as
  text): what put_run stores and query returns, so a round trip through any store gives back exactly this."""
  run = rec.get("run") or {}
  extra = run.get("extra")
  out_run = {k: _str(run.get(k)) for k in RUN_FIELDS if k != "extra"}
  out_run["extra"] = json.loads(json.dumps(extra, sort_keys=True)) if isinstance(extra, dict) else {}
  gates = []
  for g in rec.get("gates") or []:
    x = {k: _str(g.get(k)) for k in GATE_FIELDS if k not in GATE_NUMBERS}
    x.update({k: _num(g.get(k)) for k in GATE_NUMBERS})
    gates.append({k: x[k] for k in GATE_FIELDS})
  metrics = [{"name": str(m.get("name")), "profile": str(m.get("profile") or ""), "concurrency": int(m.get("concurrency") or 0), "value": _num(m.get("value"))}
             for m in rec.get("metrics") or []]
  return {"record_version": int(rec.get("record_version") or RECORD_VERSION), "run": out_run,
          "gates": sorted(gates, key=lambda g: g["gate_id"]), "metrics": sorted(metrics, key=lambda m: (m["name"], m["profile"], m["concurrency"]))}

def invarix_record(rec: Any) -> dict:
  """The normalized record, or StoreError saying what is wrong with it."""
  if not isinstance(rec, dict) or not isinstance(rec.get("run"), dict): raise StoreError("a record is {record_version, run: {...}, gates: [...], metrics: [...]}")
  v = rec.get("record_version", RECORD_VERSION)
  if v != RECORD_VERSION: raise StoreError(f"record_version {v!r}: this store reads record version {RECORD_VERSION}")
  key = rec["run"].get("key")
  if not isinstance(key, str) or not _KEY.match(key): raise StoreError(f"run.key must be the sha256 of the report file (64 lowercase hex characters), not {key!r}")
  if not rec["run"].get("group_key"): raise StoreError("run.group_key is required")
  if not isinstance(rec.get("gates") or [], list) or not isinstance(rec.get("metrics") or [], list): raise StoreError("gates and metrics are lists")
  n = normalize_record(rec)
  ids = [g["gate_id"] for g in n["gates"]]
  if any(not i or i == "None" for i in ids) or len(set(ids)) != len(ids): raise StoreError("every gate needs a gate_id, unique within the run")
  mk = [(m["name"], m["profile"], m["concurrency"]) for m in n["metrics"]]
  if len(set(mk)) != len(mk): raise StoreError("metrics must be unique by (name, profile, concurrency)")
  return n
