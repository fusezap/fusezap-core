"""The shared SQL store: tables, migrations and queries written once; a dialect sets what differs between databases
(the placeholder, the column types, insert-if-absent, how a transaction begins and how a statement runs).

Tables (prefixed fz_ so they can share a database with anything else):
  fz_schema_version (version, applied_utc)           one row per migration applied
  fz_runs (run_key primary key, record_version, the run fields, extra as JSON text)
  fz_gates (run_key, gate_id) primary key            one row per gate of a run
  fz_metrics (run_key, name, profile, concurrency) primary key

MIGRATIONS is a list of (version, [statement templates]); migrate() applies those newer than the database's version in
order, each in a transaction where the database allows DDL in one, and refuses a database whose version is newer than
this code's (SCHEMA_VERSION). Add a migration by appending (2, [...]); never edit an applied one.
"""
from __future__ import annotations
import datetime, json
from typing import Any
from fusezap.invarix_stores.contract import (GATE_FIELDS, GATE_NUMBERS, METRIC_FIELDS, RECORD_VERSION, RUN_FIELDS, STORE_API, Store, StoreError,
                                             invarix_record)

RUN_COLUMNS = ["run_key", "record_version"] + [k for k in RUN_FIELDS if k != "key"]
GATE_COLUMNS = ["run_key", "gate_id", "check_name"] + [k for k in GATE_FIELDS if k not in ("gate_id", "check")]
METRIC_COLUMNS = ["run_key"] + list(METRIC_FIELDS)
LONG_TEXT = {"name", "label", "hardware", "group_key", "baseline", "anchor", "source", "extra", "metric", "threshold"}

def _run_cols(t: dict) -> str:
  cols = [f"run_key {t['key']} NOT NULL PRIMARY KEY", "record_version INTEGER NOT NULL"]
  cols += [f"{c} {t['text'] if c in LONG_TEXT else t['str']}" for c in RUN_COLUMNS[2:]]
  return ", ".join(cols)

def _gate_cols(t: dict) -> str:
  cols = [f"run_key {t['key']} NOT NULL", f"gate_id {t['key']} NOT NULL"]
  cols += [f"{c} {t['float'] if c in GATE_NUMBERS else (t['text'] if c in LONG_TEXT else t['str'])}" for c in GATE_COLUMNS[2:]]
  return ", ".join(cols) + ", PRIMARY KEY (run_key, gate_id)"

MIGRATIONS: list[tuple[int, list[str]]] = [
  (1, ["CREATE TABLE fz_runs ({run_cols})",
       "CREATE TABLE fz_gates ({gate_cols})",
       "CREATE TABLE fz_metrics (run_key {key} NOT NULL, name {key} NOT NULL, profile {key} NOT NULL, concurrency INTEGER NOT NULL, "
       "value {float}, PRIMARY KEY (run_key, name, profile, concurrency))",
       "CREATE INDEX fz_runs_finished ON fz_runs (finished_utc)",
       "CREATE INDEX fz_runs_model ON fz_runs (model)"]),
]
SCHEMA_VERSION = MIGRATIONS[-1][0]

class SQLStore(Store):
  """A store on a DB-API 2 connection. A backend subclasses it with its dialect and opens `self.conn`."""
  api = STORE_API
  scheme = ""
  placeholder = "?"                       # "?" (qmark) or "%s" (format)
  types = {"key": "VARCHAR(191)", "str": "VARCHAR(255)", "text": "TEXT", "float": "DOUBLE PRECISION"}
  transactional_ddl = True                 # MySQL commits DDL implicitly: migrations there are applied statement by statement

  def __init__(self, conn: Any, label: str = ""):
    self.conn, self.label = conn, label

  # ---- dialect hooks
  def execute(self, sql: str, params: tuple | list = ()) -> list[tuple]:
    """Run one statement; the rows of a SELECT, else []."""
    cur = self.conn.cursor()
    try:
      cur.execute(sql, tuple(params))
      return list(cur.fetchall()) if cur.description else []
    finally: cur.close()
  def begin(self) -> None: pass            # DB-API: a transaction is open after connect / commit
  def commit(self) -> None: self.conn.commit()
  def rollback(self) -> None: self.conn.rollback()
  def insert_ignore(self, table: str, cols: list[str], keys: list[str]) -> str:
    """INSERT that skips a row whose primary key exists (SQL standard-ish ON CONFLICT; MySQL overrides)."""
    return f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join([self.placeholder] * len(cols))}) ON CONFLICT ({', '.join(keys)}) DO NOTHING"
  def table_exists(self, name: str) -> bool:
    return bool(self.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = {self.placeholder}", (name,))[0][0])

  # ---- the contract
  def _q(self, sql: str) -> str: return sql.replace("?", self.placeholder)

  def schema_version(self) -> int:
    if not self.table_exists("fz_schema_version"): return 0
    rows = self.execute("SELECT MAX(version) FROM fz_schema_version")
    return int(rows[0][0] or 0) if rows else 0

  def migrate(self) -> int:
    t = {**self.types, "run_cols": _run_cols(self.types), "gate_cols": _gate_cols(self.types)}
    try:
      self.begin()
      if not self.table_exists("fz_schema_version"):
        self.execute(f"CREATE TABLE fz_schema_version (version INTEGER NOT NULL PRIMARY KEY, applied_utc {self.types['str']})")
      self.commit()
      cur = self.schema_version()
      if cur > SCHEMA_VERSION:
        raise StoreError(f"this {self.scheme} database has history schema version {cur}, newer than this code's {SCHEMA_VERSION}: upgrade fusezap")
      for v, stmts in MIGRATIONS:
        if v <= cur: continue
        self.begin()
        for s in stmts: self.execute(s.format(**t))
        self.execute(self._q("INSERT INTO fz_schema_version (version, applied_utc) VALUES (?, ?)"),
                     (v, datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
        self.commit()
      return SCHEMA_VERSION
    except Exception:
      try: self.rollback()
      except Exception: pass
      raise

  def put_run(self, record: dict) -> bool:
    rec = invarix_record(record)
    run = rec["run"]
    if self.execute(self._q("SELECT 1 FROM fz_runs WHERE run_key = ?"), (run["key"],)): return False
    row = [run["key"], rec["record_version"]] + [json.dumps(run["extra"], sort_keys=True) if k == "extra" else run[k] for k in RUN_FIELDS if k != "key"]
    try:
      self.begin()
      self.execute(self.insert_ignore("fz_runs", RUN_COLUMNS, ["run_key"]), row)
      gi = self.insert_ignore("fz_gates", GATE_COLUMNS, ["run_key", "gate_id"])
      for g in rec["gates"]: self.execute(gi, [run["key"]] + [g[k] for k in GATE_FIELDS])
      mi = self.insert_ignore("fz_metrics", METRIC_COLUMNS, ["run_key", "name", "profile", "concurrency"])
      for m in rec["metrics"]: self.execute(mi, [run["key"]] + [m[k] for k in METRIC_FIELDS])
      self.commit()
    except Exception:
      self.rollback(); raise
    return True

  def keys(self) -> set[str]:
    return {r[0] for r in self.execute("SELECT run_key FROM fz_runs")}

  def query(self, group_key: str | None = None, model: str | None = None, since: str | None = None, limit: int | None = None) -> list[dict]:
    where, params = [], []
    for col, v in (("group_key", group_key), ("model", model)):
      if v is not None: where.append(f"{col} = ?"); params.append(v)
    if since is not None: where.append("finished_utc >= ?"); params.append(since)
    sql = f"SELECT {', '.join(RUN_COLUMNS)} FROM fz_runs" + (" WHERE " + " AND ".join(where) if where else "")
    rows = sorted(self.execute(self._q(sql), params), key=lambda r: (r[RUN_COLUMNS.index("finished_utc")] or "", r[0]))   # NULLs first on every database
    if limit is not None: rows = rows[-int(limit):] if limit > 0 else []
    out = []
    for r in rows:
      d = dict(zip(RUN_COLUMNS, r))
      run = {"key": d["run_key"], **{k: d[k] for k in RUN_FIELDS if k not in ("key", "extra")}}
      run["extra"] = json.loads(d["extra"]) if d.get("extra") else {}
      gates = [dict(zip(GATE_FIELDS, g[1:])) for g in
               self.execute(self._q(f"SELECT {', '.join(GATE_COLUMNS)} FROM fz_gates WHERE run_key = ? ORDER BY gate_id"), (d["run_key"],))]
      metrics = [dict(zip(METRIC_FIELDS, m[1:])) for m in
                 self.execute(self._q(f"SELECT {', '.join(METRIC_COLUMNS)} FROM fz_metrics WHERE run_key = ? ORDER BY name, profile, concurrency"), (d["run_key"],))]
      out.append(invarix_record({"record_version": int(d["record_version"] or RECORD_VERSION), "run": run, "gates": gates, "metrics": metrics}))
    return out

  def close(self) -> None:
    if self.conn is not None:
      try: self.conn.close()
      finally: self.conn = None
