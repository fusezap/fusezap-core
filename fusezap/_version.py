# The single source of truth for the FuseZap Core version. pyproject.toml reads it (hatch `[tool.hatch.version]`),
# `fusezap.__version__` imports it, and scripts/release.py rewrites it on `cut`. PEP 440: X.Y.Z, X.Y.ZrcN, X.Y.Z.devN.
__version__ = "0.1.0.dev0"
