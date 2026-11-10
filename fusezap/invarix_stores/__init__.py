"""History stores for FuseZap Invarix release history: the store contract, the record schema and the database backends.

This package is open source (Apache-2.0, like the rest of FuseZap Core; LICENSE and NOTICE at the repository root). It
imports nothing from the licensed Invarix code: Invarix turns its report files into records (schema below) and hands
them to a store; a store only keeps and returns records. The report files stay the source of truth; a store is an
index that can be rebuilt from them at any time, and a run is keyed by the sha256 of its report file, so importing
the same report twice changes nothing.

  from fusezap.invarix_stores import open_store
  with open_store("sqlite:///history.db") as st:      # opens, then migrates the schema to the current version
    st.put_run(record)                                  # True: stored; False: that run (key) was already there
    runs = st.query(group_key=g)                        # records, oldest first

Built-in backends, chosen by the URL scheme: sqlite (standard library), duckdb, postgresql (psycopg 3) and mysql
(PyMySQL). The drivers are optional extras: pip install 'fusezap[invarix-duckdb]', 'fusezap[invarix-postgres]',
'fusezap[invarix-mysql]'. A missing driver raises MissingDriver, whose message names the extra. Other databases are
plugins: a Python package that registers a factory in the entry-point group `fusezap.invarix.stores` (WRITING_A_STORE.md
in this directory; the built-ins are registered the same way). The contract is versioned: STORE_API.

A store connects only to the database its URL names. Passwords in a URL are never printed: redact_url().
"""
from fusezap.invarix_stores.contract import (STORE_API, RECORD_VERSION, Store, StoreError, MissingDriver, invarix_record,  # noqa: F401
                                             normalize_record)
from fusezap.invarix_stores.registry import ENTRY_POINT_GROUP, BUILTIN, open_store, schemes, redact_url, scheme_of  # noqa: F401
