"""Vendored BFCL execution environment. See NOTICE.

The vendored files import each other by their ORIGINAL package path
(bfcl_eval.eval_checker.multi_turn_eval.func_source_code.long_context). Rather than
rewrite those imports -- which would make a future re-vendor a merge instead of a file
copy -- this module registers that path as an alias for this package before any submodule
is loaded. The files therefore stay byte-identical to upstream.
"""
from __future__ import annotations

import importlib
import sys

_ALIAS = "bfcl_eval.eval_checker.multi_turn_eval.func_source_code"


def _install_alias() -> None:
    if _ALIAS in sys.modules:
        return
    import types
    parts = _ALIAS.split(".")
    for i in range(1, len(parts)):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []          # mark as a package so submodule import works
            sys.modules[name] = mod
    sys.modules[_ALIAS] = sys.modules[__name__]
    # the alias package must resolve submodules to THIS directory
    sys.modules[_ALIAS].__path__ = __path__


_install_alias()

# long_context is imported by several of the API modules; load it under the aliased name
# so those imports resolve to the same module object rather than a second copy.
importlib.import_module(_ALIAS + ".long_context")
