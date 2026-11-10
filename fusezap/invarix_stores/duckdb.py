"""duckdb:///history.duckdb (relative path), duckdb:////abs/path/history.duckdb, duckdb:///:memory:.
Needs the duckdb package: pip install 'fusezap[invarix-duckdb]'."""
from __future__ import annotations
import os
from fusezap.invarix_stores.registry import need
from fusezap.invarix_stores.sql import SQLStore
from fusezap.invarix_stores.sqlite import db_path

class DuckDBStore(SQLStore):
  scheme = "duckdb"
  placeholder = "?"
  types = {"key": "VARCHAR", "str": "VARCHAR", "text": "VARCHAR", "float": "DOUBLE"}
  def execute(self, sql: str, params=()) -> list[tuple]:
    # a DuckDB cursor() is a second connection with its own transaction: run on the connection itself
    res = self.conn.execute(sql, list(params))
    return list(res.fetchall()) if sql.lstrip().upper().startswith("SELECT") else []
  def begin(self) -> None: self.conn.begin()
  def commit(self) -> None: self.conn.commit()
  def rollback(self) -> None:
    try: self.conn.rollback()
    except Exception: pass          # no transaction open
  def table_exists(self, name: str) -> bool:
    return bool(self.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", (name,))[0][0])

def open(url: str) -> DuckDBStore:
  duckdb = need("duckdb", "duckdb", "invarix-duckdb", "duckdb")
  path = db_path(url)
  if path != ":memory:" and os.path.dirname(os.path.abspath(path)): os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  return DuckDBStore(duckdb.connect(path), os.path.basename(path))
