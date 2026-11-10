"""postgresql://user:password@host:5432/dbname (also postgres://; any libpq URI parameter, e.g. ?sslmode=require).
Needs psycopg 3: pip install 'fusezap[invarix-postgres]'. Amazon RDS / Aurora PostgreSQL, Cloud SQL, Azure Database
for PostgreSQL and other PostgreSQL-compatible services work through this backend. Put the URL (it holds the password)
in an environment variable, not on a command line."""
from __future__ import annotations
from fusezap.invarix_stores.registry import need, redact_url
from fusezap.invarix_stores.sql import SQLStore

class PostgresStore(SQLStore):
  scheme = "postgresql"
  placeholder = "%s"
  types = {"key": "VARCHAR(191)", "str": "VARCHAR(255)", "text": "TEXT", "float": "DOUBLE PRECISION"}
  def table_exists(self, name: str) -> bool:
    return bool(self.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = current_schema() AND table_name = %s", (name,))[0][0])

def open(url: str) -> PostgresStore:
  psycopg = need("psycopg", "postgresql", "invarix-postgres", "psycopg[binary]")
  if url.startswith("postgres://"): url = "postgresql://" + url[len("postgres://"):]
  return PostgresStore(psycopg.connect(url, connect_timeout=15), redact_url(url))
