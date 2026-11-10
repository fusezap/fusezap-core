# Writing a history store plugin

FuseZap Invarix keeps its release history in report files; a *store* is an optional database index of those runs.
The built-in stores are sqlite, duckdb, postgresql and mysql (this package, Apache-2.0). Any other database (BigQuery,
Snowflake, ClickHouse, a company's own service) is a plugin: a small Python package that implements the store contract
and registers itself. Invarix finds it by the URL scheme the user passes (`--db` / `--db-env`); nothing else changes.

## The contract (STORE_API = 1)

FuseZap Invarix is the whole contract. A store is an object with:

| member | does |
|---|---|
| `api = 1` | the contract version it implements (`STORE_API`); a mismatch is refused with a clear message |
| `scheme` | the URL scheme it serves, e.g. `"bigquery"` |
| `migrate() -> int` | create or upgrade its tables to its current schema; safe to call every time |
| `put_run(record) -> bool` | store one run; `False`, and no change, when a run with that `run.key` is already stored |
| `query(group_key=None, model=None, since=None, limit=None) -> list[record]` | runs oldest first (`finished_utc`, then key); `limit`: the newest N, still oldest first |
| `keys() -> set[str]` | the run keys stored |
| `close()` | release the connection (`Store` makes it a context manager) |

A record is plain data: `{record_version: 1, run: {...}, gates: [...], metrics: [...]}` (fields in contract.py).
`run.key` is the sha256 of the run's report file: a run's identity, so importing a report again is a no-op. Call
`invarix_record(record)` first: it checks the record and returns it in canonical form (numbers as floats, the gate
threshold as text); `query` must return records in that same form, so a put / query round trip is exact.

## A SQL database: reuse the shared base

For a database with a DB-API 2 driver, subclass `fusezap.invarix_stores.sql.SQLStore` and set only what differs:

```python
# acme_history_store/__init__.py
from fusezap.invarix_stores.registry import need, redact_url
from fusezap.invarix_stores.sql import SQLStore

class AcmeStore(SQLStore):
    scheme = "acmedb"
    placeholder = "%s"                     # "?" (qmark) or "%s" (format)
    types = {"key": "VARCHAR(191)", "str": "VARCHAR(255)", "text": "TEXT", "float": "DOUBLE"}
    def insert_ignore(self, table, cols, keys):   # only if the database lacks INSERT ... ON CONFLICT DO NOTHING
        return f"INSERT IGNORE INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})"

def open(url):
    acmedb = need("acmedb", "acmedb", "acme", "acmedb-driver")   # a missing driver says how to install it
    return AcmeStore(acmedb.connect(url), redact_url(url))
```

The tables (`fz_runs`, `fz_gates`, `fz_metrics`, `fz_schema_version`), the migrations and the queries come from the
base. The other hooks, for databases that need them: `execute` (how one statement runs and returns rows),
`begin` / `commit` / `rollback`, and `table_exists`. See sqlite.py, duckdb.py, postgresql.py and mysql.py in this
directory: each is a few lines.

## Anything else: implement the five methods

A warehouse or an HTTP service implements `migrate`, `put_run`, `query`, `keys` and `close` itself (subclass
`fusezap.invarix_stores.Store` for the context manager). Keep `put_run` idempotent by key and `query` ordered.

## Register it

In the plugin's `pyproject.toml`, the entry-point name is the URL scheme and the value the factory:

```toml
[project.entry-points."fusezap.invarix.stores"]
acmedb = "acme_history_store:open"
```

After `pip install acme-history-store`, `--db acmedb://...` uses it. A plugin cannot take over a built-in scheme.
The built-ins are registered the same way in FuseZap's own pyproject.toml.

## Test it with the shared contract suite

`tests/test_invarix_stores.py` is the suite every built-in store passes. Add the plugin's URL factory to `BACKENDS`
(or copy the suite into the plugin's repository and point it at a throwaway database) and run it: migrate idempotence,
put-once-by-key, an exact round trip, the query filters, refusal of a newer schema, and bad records changing nothing.

## Rules every store keeps

- Connect only to the database in the URL; send nothing anywhere else.
- Never print or log a password: use `redact_url(url)` in messages (`open_store` also scrubs exceptions).
- The URL with a password comes from an environment variable (`--db-env VAR`), never a command line.
- The database is an index: it can be dropped and rebuilt from the report files at any time, so never make it the
  only copy of anything.
- Schema changes are new migrations appended to the list, never edits of an applied one; refuse a database whose
  schema is newer than the code.
