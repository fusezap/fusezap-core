"""sqlite:///history.db (a path relative to the current directory), sqlite:////abs/path/history.db, sqlite:///:memory:.
Python's standard library; no extra needed."""
from __future__ import annotations
import os, sqlite3
from urllib.parse import urlsplit
from fusezap.invarix_stores.contract import StoreError
from fusezap.invarix_stores.sql import SQLStore

def db_path(url: str) -> str:
  """The file path of a sqlite:// or duckdb:// URL: one slash after the scheme's // starts a relative path, two an absolute one."""
  p = urlsplit(url)
  if p.netloc not in ("", "localhost"): raise StoreError(f"{p.scheme}:// takes a file path (sqlite:///history.db), not a host")
  path = p.path[1:] if p.path.startswith("/") else p.path
  if not path: raise StoreError(f"{p.scheme}:// needs a file path, e.g. {p.scheme}:///history.db")
  return path

class SQLiteStore(SQLStore):
  scheme = "sqlite"
  placeholder = "?"
  types = {"key": "TEXT", "str": "TEXT", "text": "TEXT", "float": "REAL"}
  def begin(self) -> None:
    if not self.conn.in_transaction: self.conn.execute("BEGIN")
  def table_exists(self, name: str) -> bool:
    return bool(self.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?", (name,))[0][0])

def open(url: str) -> SQLiteStore:
  path = db_path(url)
  if path != ":memory:" and os.path.dirname(os.path.abspath(path)): os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  conn = sqlite3.connect(path, isolation_level=None)   # autocommit; begin() / commit() bracket each write
  conn.execute("PRAGMA foreign_keys = ON")
  return SQLiteStore(conn, os.path.basename(path))
