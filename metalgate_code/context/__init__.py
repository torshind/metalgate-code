"""Context package for Python symbol indexing and storage.

This package provides:
  - Symbol extraction from Python modules (context.parsing)
  - Database storage and querying (context.db)
  - High-level orchestration (context.indexer)

Main exports:
  IndexStore: Database query interface for symbols.
  build_index: Build the symbol index from site-packages.
"""

from metalgate_code.context.data import (
    _AttrData,
    _ClassData,
    _DecoratorApp,
    _FuncData,
    _ModuleData,
    _ParamData,
)
from metalgate_code.context.db import (
    Attribute,
    Base,
    Class,
    Function,
    IndexStore,
    Module,
    Package,
    Parameter,
    write_index,
)
from metalgate_code.context.indexer import build_index
from metalgate_code.context.parsing import collect_files, find_site_packages, parse_file
from metalgate_code.context.resolver import _resolve_forwarding

__all__ = [
    # Database models
    "Base",
    "Package",
    "Module",
    "Function",
    "Parameter",
    "Class",
    "Attribute",
    # Store and writer
    "IndexStore",
    "write_index",
    # Transient data structures
    "_ModuleData",
    "_FuncData",
    "_ClassData",
    "_AttrData",
    "_ParamData",
    "_DecoratorApp",
    # Extraction functions
    "find_site_packages",
    "collect_files",
    "parse_file",
    "_resolve_forwarding",
    # Build index
    "build_index",
]
