"""Context package for Python symbol indexing and storage.

This package provides:
  - Symbol extraction from Python modules (context.parsing)
  - Database storage and querying (context.db)
  - High-level orchestration (context.indexer)

Main exports:
  IndexStore: Database query interface for symbols.
  StreamingWriter: Async incremental index builder.
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
    Module,
    Package,
    Parameter,
    StreamingWriter,
    _IndexStore,
)
from metalgate_code.context.indexer import (
    IndexStore,
    is_indexing,
    start_indexing,
)
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
    "_IndexStore",
    "StreamingWriter",
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
    # Indexer
    "start_indexing",
    "is_indexing",
    "IndexStore",
]
