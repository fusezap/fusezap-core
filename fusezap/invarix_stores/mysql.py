"""mysql://user:password@host:3306/dbname (?ssl=true, or ?ssl_ca=/path/ca.pem for a CA file). Needs PyMySQL:
pip install 'fusezap[invarix-mysql]'. MySQL 8 and MariaDB 10.5+; Amazon RDS / Aurora MySQL, Cloud SQL for MySQL and
Azure Database for MySQL work through this backend. Put the URL (it holds the password) in an environment variable."""
from __future__ import annotations
from urllib.parse import parse_qsl, unquote, urlsplit
from fusezap.invarix_stores.contract import StoreError
from fusezap.invarix_stores.registry import need, redact_url
from fusezap.invarix_stores.sql import SQLStore

class MySQLStore(SQLStore):
  scheme = "mysql"
  placeholder = "%s"
  types = {"key": "VARCHAR(191)", "str": "VARCHAR(255)", "text": "TEXT", "float": "DOUBLE"}
  transactional_ddl = False
  def insert_ignore(self, table: str, cols: list[str], keys: list[str]) -> str:
    return f"INSERT IGNORE INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})"
  def table_exists(self, name: str) -> bool:
    return bool(self.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = %s", (name,))[0][0])

def connect_args(url: str) -> dict:
  """PyMySQL connect() keyword arguments from a mysql:// URL."""
  p = urlsplit(url)
  db = unquote(p.path.lstrip("/"))
  if not p.hostname or not db: raise StoreError(f"{redact_url(url)}: a MySQL URL is mysql://user:password@host:3306/dbname")
  q = dict(parse_qsl(p.query))
  args = {"host": p.hostname, "port": p.port or 3306, "user": unquote(p.username or ""), "password": unquote(p.password or ""), "database": db,
          "charset": "utf8mb4", "connect_timeout": 15, "autocommit": False}
  if q.get("ssl_ca"): args["ssl"] = {"ca": q["ssl_ca"]}
  elif q.get("ssl", "").lower() in ("1", "true", "required", "require"): args["ssl"] = {}
  return args

def open(url: str) -> MySQLStore:
  pymysql = need("pymysql", "mysql", "invarix-mysql", "PyMySQL")
  return MySQLStore(pymysql.connect(**connect_args(url)), redact_url(url))
