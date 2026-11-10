"""The history-store contract (fusezap/invarix_stores): one suite every backend must pass.

sqlite (standard library) and duckdb (when installed) run everywhere. PostgreSQL and MySQL run against a server named
by FZ_TEST_POSTGRES_URL / FZ_TEST_MYSQL_URL (a throwaway database: the suite drops its fz_ tables first), or, with
FZ_TEST_STORE_CONTAINERS=1 and docker on the PATH, against containers it starts (postgres:16, mysql:8.4) and removes;
otherwise they are skipped with that reason. A new backend adds its URL factory to BACKENDS.
"""
from __future__ import annotations
import ast, copy, json, os, re, shutil, socket, subprocess, sys, time, uuid
import pytest

from fusezap import invarix_stores as S
from fusezap.invarix_stores import contract as K, registry as REG, sql as SQL

PKG = os.path.dirname(S.__file__)

def _record(i: int, group: str = "g1", model: str = "org/model-a", finished: str | None = None) -> dict:
  key = f"{i:064x}"
  return {"record_version": 1,
          "run": {"key": key, "run_id": f"run{i}", "name": f"release {i}", "label": f"rel {i}", "started_utc": None,
                  "finished_utc": finished or f"2026-10-{i + 1:02d}T00:00:00Z", "tool_version": "0.2.0", "mode": "stored", "engine": "fusezap",
                  "engine_version": None, "model": model, "hardware": "2x H200", "dtype": "bf16", "workload": "short", "load_points": "1,8,32",
                  "group_key": group, "verdict": "PASS", "decision": "PROMOTE", "baseline": "rel 0", "anchor": None, "source": f"rel{i}/report.json",
                  "extra": {"note": "x", "n": i}},
          "gates": [{"gate_id": "relative.throughput@8", "check": "performance", "kind": "relative", "against": None, "status": "pass",
                     "metric": "performance.points[concurrency=8].output_tok_s", "comparator": ">=", "threshold": 0.97, "value": 1.01 + i / 100,
                     "baseline_value": 250.0, "ci_lo": 0.99, "ci_hi": 1.03},
                    {"gate_id": "functional.usage", "check": "functional", "kind": "absolute", "status": "pass", "threshold": "pass", "value": None},
                    {"gate_id": "anchor.ttft_p99@8", "check": "performance", "kind": "relative", "against": "anchor", "status": "fail", "threshold": 1.2,
                     "value": 1.31}],
          "metrics": [{"name": "output_tok_s", "concurrency": c, "profile": "", "value": 100.0 * c + i} for c in (1, 8, 32)] +
                     [{"name": "top1_agreement", "value": 0.995}, {"name": "error_rate", "value": 0}]}

# ---------------------------------------------------------------- backends
def _free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0)); return s.getsockname()[1]

def _container(image: str, port: int, env: dict, url: str, ready) -> str | None:
  """Start a throwaway database container; its URL once it answers (None on failure)."""
  name = f"fz-store-test-{uuid.uuid4().hex[:8]}"
  args = ["docker", "run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{_free_port() if port == 0 else port}:{env.pop('_PORT')}"]
  for k, v in env.items(): args += ["-e", f"{k}={v}"]
  if subprocess.run(args + [image], capture_output=True).returncode != 0: return None
  _CONTAINERS.append(name)
  for _ in range(120):
    try:
      if ready(): return url
    except Exception: pass
    time.sleep(1)
  return None

_CONTAINERS: list[str] = []

def _server_url(kind: str) -> tuple[str | None, str]:
  """(URL, why not): FZ_TEST_<KIND>_URL, else a container when FZ_TEST_STORE_CONTAINERS=1 and docker exists."""
  env = os.environ.get(f"FZ_TEST_{kind.upper()}_URL")
  if env: return env, ""
  if os.environ.get("FZ_TEST_STORE_CONTAINERS") != "1":
    return None, (f"no {kind} server: set FZ_TEST_{kind.upper()}_URL to a throwaway database, or FZ_TEST_STORE_CONTAINERS=1 with docker "
                  "to start one in a container")
  if not shutil.which("docker"): return None, f"FZ_TEST_STORE_CONTAINERS=1 but docker is not on the PATH: {kind} not tested"
  port = _free_port()
  if kind == "postgres":
    url = f"postgresql://fz:fztest@127.0.0.1:{port}/fz"
    ready = lambda: (__import__("psycopg").connect(url, connect_timeout=2).close() or True)
    got = _container("postgres:16", port, {"_PORT": 5432, "POSTGRES_USER": "fz", "POSTGRES_PASSWORD": "fztest", "POSTGRES_DB": "fz"}, url, ready)
  else:
    url = f"mysql://fz:fztest@127.0.0.1:{port}/fz"
    from fusezap.invarix_stores.mysql import connect_args
    ready = lambda: (__import__("pymysql").connect(**{**connect_args(url), "connect_timeout": 2}).close() or True)
    got = _container("mysql:8.4", port, {"_PORT": 3306, "MYSQL_USER": "fz", "MYSQL_PASSWORD": "fztest", "MYSQL_DATABASE": "fz",
                                          "MYSQL_ROOT_PASSWORD": "fzroot"}, url, ready)
  return got, ("" if got else f"the {kind} container did not start")

def _drop_tables(url: str) -> None:
  st = S.open_store(url, migrate=False)
  try:
    for t in ("fz_metrics", "fz_gates", "fz_runs", "fz_schema_version"):
      if st.table_exists(t): st.begin(); st.execute(f"DROP TABLE {t}"); st.commit()
  finally: st.close()

def _sqlite(tmp_path): return f"sqlite:///{tmp_path / 'h.db'}"
def _duckdb(tmp_path):
  pytest.importorskip("duckdb", reason="duckdb not installed (pip install 'fusezap[invarix-duckdb]')")
  return f"duckdb:///{tmp_path / 'h.duckdb'}"
def _server(kind, module):
  def make(tmp_path):
    pytest.importorskip(module, reason=f"{module} not installed")
    url, why = _SERVERS.setdefault(kind, _server_url(kind))
    if not url: pytest.skip(why)
    _drop_tables(url)
    return url
  return make
_SERVERS: dict[str, tuple] = {}

BACKENDS = {"sqlite": _sqlite, "duckdb": _duckdb, "postgresql": _server("postgres", "psycopg"), "mysql": _server("mysql", "pymysql")}

@pytest.fixture(scope="module", autouse=True)
def _remove_containers():
  yield
  for n in _CONTAINERS: subprocess.run(["docker", "rm", "-f", n], capture_output=True)

@pytest.fixture(params=list(BACKENDS))
def url(request, tmp_path):
  return BACKENDS[request.param](tmp_path)

# ---------------------------------------------------------------- the contract, every backend
def test_migrate_creates_the_schema_once(url):
  with S.open_store(url) as st:
    assert st.api == S.STORE_API and st.schema_version() == SQL.SCHEMA_VERSION
    assert st.migrate() == SQL.SCHEMA_VERSION and st.keys() == set()        # again: nothing to do
    assert st.execute("SELECT COUNT(*) FROM fz_schema_version")[0][0] == len(SQL.MIGRATIONS)

def test_put_run_is_idempotent_by_key(url):
  rec = _record(1)
  with S.open_store(url) as st:
    assert st.put_run(rec) is True
    assert st.put_run(rec) is False                                           # the same report imported again
    changed = copy.deepcopy(rec); changed["run"]["verdict"] = "FAIL"
    assert st.put_run(changed) is False and st.query()[0]["run"]["verdict"] == "PASS"   # a key is immutable
    assert st.keys() == {rec["run"]["key"]}
  with S.open_store(url) as st:                                               # persisted, and the schema is not redone
    assert st.keys() == {rec["run"]["key"]} and st.put_run(rec) is False

def test_round_trip_is_exact(url):
  recs = [_record(i) for i in (3, 1, 2)]
  with S.open_store(url) as st:
    for r in recs: assert st.put_run(r)
    got = st.query()
  assert [g["run"]["key"] for g in got] == [r["run"]["key"] for r in sorted(recs, key=lambda r: r["run"]["finished_utc"])]
  assert got == [K.normalize_record(r) for r in sorted(recs, key=lambda r: r["run"]["finished_utc"])]
  g = {x["gate_id"]: x for x in got[0]["gates"]}
  assert g["relative.throughput@8"]["threshold"] == "0.97" and g["functional.usage"]["threshold"] == "pass" and g["anchor.ttft_p99@8"]["against"] == "anchor"
  assert got[0]["run"]["extra"] == {"n": 1, "note": "x"} and {(m["name"], m["concurrency"]) for m in got[0]["metrics"]} >= {("output_tok_s", 32), ("error_rate", 0)}

def test_query_filters(url):
  with S.open_store(url) as st:
    for i in range(1, 6): st.put_run(_record(i, group="g1" if i % 2 else "g2", model="org/model-a" if i < 4 else "org/model-b"))
    assert [r["run"]["run_id"] for r in st.query(group_key="g1")] == ["run1", "run3", "run5"]
    assert [r["run"]["run_id"] for r in st.query(model="org/model-b")] == ["run4", "run5"]
    assert [r["run"]["run_id"] for r in st.query(since="2026-10-04T00:00:00Z")] == ["run3", "run4", "run5"]
    assert [r["run"]["run_id"] for r in st.query(limit=2)] == ["run4", "run5"] and st.query(group_key="nope") == []

def test_a_newer_schema_is_refused(url):
  with S.open_store(url) as st:
    st.begin(); st.execute(st._q("INSERT INTO fz_schema_version (version, applied_utc) VALUES (?, ?)"), (SQL.SCHEMA_VERSION + 1, "2099-01-01T00:00:00Z")); st.commit()
  with pytest.raises(S.StoreError, match="newer than this code"): S.open_store(url)

def test_a_bad_record_changes_nothing(url):
  with S.open_store(url) as st:
    bad = _record(1); bad["run"]["key"] = "not-a-sha256"
    with pytest.raises(S.StoreError, match="sha256"): st.put_run(bad)
    dup = _record(2); dup["gates"].append(dict(dup["gates"][0]))
    with pytest.raises(S.StoreError, match="unique"): st.put_run(dup)
    assert st.keys() == set()

# ---------------------------------------------------------------- registry, URLs, plugins
def test_unknown_scheme_lists_the_available_ones():
  with pytest.raises(S.StoreError, match="no store for bigquery:// URLs.*sqlite"): S.open_store("bigquery://project/dataset")
  with pytest.raises(S.StoreError, match="not a store URL"): S.open_store("history.db")
  assert {"sqlite", "duckdb", "postgresql", "mysql"} <= set(S.schemes())

def test_missing_driver_names_the_extra(monkeypatch):
  real = REG.importlib.import_module
  def fake(name, *a, **k):
    if name in ("duckdb", "psycopg", "pymysql"): raise ImportError(name)
    return real(name, *a, **k)
  monkeypatch.setattr(REG.importlib, "import_module", fake)
  for url, extra in (("duckdb:///x.duckdb", "invarix-duckdb"), ("postgresql://u:pw@h/db", "invarix-postgres"), ("mysql://u:pw@h/db", "invarix-mysql")):
    with pytest.raises(S.MissingDriver, match=rf"pip install 'fusezap\[{extra}\]'"): S.open_store(url)

def test_passwords_never_shown():
  assert S.redact_url("postgresql://fz:s3cret@example.com:5432/hist?sslmode=require") == "postgresql://fz:***@example.com:5432/hist?sslmode=require"
  assert S.redact_url("mysql://fz@h/db?password=s3cret&ssl=true") == "mysql://fz@h/db?password=***&ssl=true"
  assert S.redact_url("sqlite:///h.db") == "sqlite:///h.db"
  pytest.importorskip("psycopg")
  port = _free_port()   # nothing listens there
  with pytest.raises(S.StoreError) as e: S.open_store(f"postgresql://fz:s3cret-pw@127.0.0.1:{port}/hist")
  assert "s3cret-pw" not in str(e.value) and "fz:***@" in str(e.value)

class _MemStore(K.Store):
  """A minimal plugin store kept in memory: what a third-party backend implements."""
  scheme = "mem"
  def __init__(self, url): self.runs, self.migrated = {}, 0
  def migrate(self): self.migrated += 1; return 1
  def put_run(self, record):
    r = K.invarix_record(record)
    if r["run"]["key"] in self.runs: return False
    self.runs[r["run"]["key"]] = r; return True
  def query(self, group_key=None, model=None, since=None, limit=None):
    out = sorted((r for r in self.runs.values() if (group_key is None or r["run"]["group_key"] == group_key)), key=lambda r: (r["run"]["finished_utc"] or "", r["run"]["key"]))
    return out[-limit:] if limit else out
  def keys(self): return set(self.runs)

class _EP:
  def __init__(self, name, value, obj): self.name, self.value, self._obj = name, value, obj
  def load(self): return self._obj

def test_plugins_come_from_entry_points(monkeypatch):
  class Old(_MemStore): api = 0
  eps = [_EP("mem", "acme_store:open", _MemStore), _EP("old", "acme_old:open", Old), _EP("sqlite", "evil:open", _MemStore)]
  import importlib.metadata as md
  monkeypatch.setattr(md, "entry_points", lambda group=None: [e for e in eps if group == S.ENTRY_POINT_GROUP])
  assert S.schemes()["mem"] == "acme_store:open" and S.schemes()["sqlite"] == S.BUILTIN["sqlite"]   # no take-over of a built-in
  with S.open_store("mem://") as st:
    assert isinstance(st, _MemStore) and st.migrated == 1 and st.put_run(_record(1)) and not st.put_run(_record(1))
  with pytest.raises(S.StoreError, match="store API 0"): S.open_store("old://")

def test_builtins_are_registered_as_entry_points():
  """pyproject.toml registers every built-in in the entry-point group, the same way a plugin does."""
  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  try: import tomllib
  except ImportError: pytest.skip("tomllib")
  with open(os.path.join(root, "pyproject.toml"), "rb") as f: pp = tomllib.load(f)
  assert pp["project"]["entry-points"][S.ENTRY_POINT_GROUP] == S.BUILTIN
  extras = pp["project"]["optional-dependencies"]
  assert any("duckdb" in d for d in extras["invarix-duckdb"]) and any("psycopg" in d for d in extras["invarix-postgres"]) and any("pymysql" in d.lower() for d in extras["invarix-mysql"])

def test_the_package_stands_alone():
  """Open source on its own: no module of the package imports anything outside it but the standard library and its
  optional drivers (never the licensed fusezap.invarix), so it can be published and used without it."""
  allowed = {"duckdb", "psycopg", "pymysql"} | set(sys.stdlib_module_names)
  for f in sorted(os.listdir(PKG)):
    if not f.endswith(".py"): continue
    tree = ast.parse(open(os.path.join(PKG, f)).read())
    for n in ast.walk(tree):
      names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module] if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 else []
      for m in names:
        assert m.startswith("fusezap.invarix_stores") or m.split(".")[0] in allowed, (f, m)
    assert not re.search(r"fusezap\.invarix\b(?!\.stores)", open(os.path.join(PKG, f)).read()), f   # its own names: fusezap.invarix_stores, the fusezap.invarix.stores group
