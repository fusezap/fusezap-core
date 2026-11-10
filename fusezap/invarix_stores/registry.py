"""Backends by URL scheme: the built-ins, then plugins from the entry-point group `fusezap.invarix.stores`.

An entry point's name is the URL scheme it serves and its value a factory `module:function` taking the URL and
returning a Store (contract.py). The built-ins are registered in pyproject.toml the same way; BUILTIN repeats them so a
source checkout without installed package metadata still finds them. A plugin cannot take over a built-in scheme.
"""
from __future__ import annotations
import importlib, re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from fusezap.invarix_stores.contract import STORE_API, MissingDriver, Store, StoreError

ENTRY_POINT_GROUP = "fusezap.invarix.stores"
BUILTIN = {
  "sqlite": "fusezap.invarix_stores.sqlite:open",
  "duckdb": "fusezap.invarix_stores.duckdb:open",
  "postgresql": "fusezap.invarix_stores.postgresql:open",
  "postgres": "fusezap.invarix_stores.postgresql:open",
  "mysql": "fusezap.invarix_stores.mysql:open",
}
_SECRET_PARAMS = re.compile(r"^(password|passwd|pwd|secret|token|access[_-]?key|private[_-]?key)$", re.I)

def scheme_of(url: str) -> str:
  s = urlsplit(url).scheme.lower()
  if not s: raise StoreError(f"{redact_url(url)!r} is not a store URL (scheme://...; e.g. sqlite:///history.db)")
  return s

def redact_url(url: str) -> str:
  """The URL with any password (in user:password@ or a password= / token= style query parameter) shown as ***."""
  try: p = urlsplit(url)
  except ValueError: return "<unparseable URL>"
  netloc, query = p.netloc, p.query
  if "@" in netloc:
    cred, host = netloc.rsplit("@", 1)
    if ":" in cred: netloc = cred.split(":", 1)[0] + ":***@" + host
  pairs = parse_qsl(query, keep_blank_values=True)
  if any(_SECRET_PARAMS.match(k) for k, _ in pairs):
    query = urlencode([(k, "***" if _SECRET_PARAMS.match(k) else v) for k, v in pairs], safe="*/")
  if (netloc, query) == (p.netloc, p.query): return url
  return urlunsplit((p.scheme, netloc, p.path, query, p.fragment))

def password_of(url: str) -> str | None:
  try: return urlsplit(url).password
  except ValueError: return None

def scrub(text: str, url: str) -> str:
  """`text` (an error message) with the URL's password and secret parameters removed."""
  secrets = [password_of(url)] + [v for k, v in parse_qsl(urlsplit(url).query) if _SECRET_PARAMS.match(k)]
  for s in secrets:
    if s: text = text.replace(s, "***")
  return text

def _plugins() -> dict[str, object]:
  from importlib.metadata import entry_points
  out: dict[str, object] = {}
  try: eps = entry_points(group=ENTRY_POINT_GROUP)
  except Exception: return out          # broken metadata of some other package: the built-ins still work
  for ep in sorted(eps, key=lambda e: (e.name, e.value)):
    name = ep.name.lower()
    if name in BUILTIN and ep.value != BUILTIN[name]: continue   # a plugin cannot take over a built-in scheme
    out.setdefault(name, ep)
  return out

def schemes() -> dict[str, str]:
  """{scheme: "module:factory"}: every backend available (built-ins and installed plugins)."""
  out = dict(BUILTIN)
  for name, ep in _plugins().items(): out.setdefault(name, ep.value)
  return dict(sorted(out.items()))

def _load(spec: str):
  mod, _, fn = spec.partition(":")
  return getattr(importlib.import_module(mod), fn or "open")

def factory(scheme: str):
  if scheme in BUILTIN: return _load(BUILTIN[scheme])
  ep = _plugins().get(scheme)
  if ep is None:
    raise StoreError(f"no store for {scheme}:// URLs (available: {', '.join(schemes())}; a plugin registers one in the "
                     f"{ENTRY_POINT_GROUP!r} entry-point group)")
  return ep.load()

def open_store(url: str, migrate: bool = True) -> Store:
  """The store for `url`, its schema migrated to the current version (migrate=False: as it is). Raises StoreError (or
  MissingDriver) with the URL's password removed from the message."""
  scheme = scheme_of(url)
  make = factory(scheme)
  try: st = make(url)
  except StoreError as e: raise type(e)(scrub(str(e), url)) from None
  except Exception as e: raise StoreError(f"{redact_url(url)}: {type(e).__name__}: {scrub(str(e), url)}") from None
  api = getattr(st, "api", None)
  if api != STORE_API:
    try: st.close()
    finally: raise StoreError(f"the {scheme} store implements store API {api!r}; this version of the history needs API {STORE_API}")
  if migrate:
    try: st.migrate()
    except StoreError as e:
      st.close(); raise type(e)(scrub(str(e), url)) from None
    except Exception as e:
      st.close(); raise StoreError(f"{redact_url(url)}: migrating the schema: {type(e).__name__}: {scrub(str(e), url)}") from None
  return st

def need(module: str, scheme: str, extra: str, package: str):
  """Import a backend's driver, or MissingDriver naming the extra that installs it."""
  try: return importlib.import_module(module)
  except ImportError:
    raise MissingDriver(f"the {scheme} store needs the {package} package: pip install 'fusezap[{extra}]' (or pip install {package})") from None
