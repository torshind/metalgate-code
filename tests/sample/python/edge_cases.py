"""Edge case file for tracer testing."""

import functools
from typing import Any

# Top-level constant
CONSTANT = 42


# Decorated function
@functools.lru_cache(maxsize=128)
def cached_function(x: int) -> int:
    """A decorated function."""
    return x * 2


# Async function with nested function
async def async_with_nested(url: str) -> dict[str, Any]:
    """Async function containing a nested function."""
    result = await fetch(url)

    def process_result(data: dict) -> str:
        """Nested function inside async."""
        return str(data.get("key", ""))

    return {"processed": process_result(result)}


# Class with property, classmethod, staticmethod
class EdgeClass:
    """Class with various method types."""

    _instance_count = 0

    def __init__(self, value: int):
        self._value = value
        EdgeClass._instance_count += 1

    @property
    def value(self) -> int:
        """Property getter."""
        return self._value

    @classmethod
    def get_count(cls) -> int:
        """Class method."""
        return cls._instance_count

    @staticmethod
    def static_helper(x: int) -> int:
        """Static method."""
        return x + 1

    async def async_method(self) -> str:
        """Async method calling other methods."""
        count = self.get_count()
        helper = self.static_helper(count)
        return f"{self.value}-{helper}"


# Stub function (should be detected as stub)
def stub_function(x: int) -> int:
    """A stub that raises NotImplementedError."""
    raise NotImplementedError


# Another stub style
def pass_function(x: int) -> None:
    """A stub using pass."""
    pass


# Ellipsis stub
def ellipsis_function(x: int) -> None:
    """A stub using ellipsis."""
    ...


# Function with no calls
def pure_function(a: int, b: int) -> int:
    """No function calls, just arithmetic."""
    return a + b


# Function calling stdlib only
def stdlib_only(data: list) -> int:
    """Only calls stdlib functions."""
    return len(data)


# Awaited coroutine placeholder
async def fetch(url: str) -> dict:
    """Placeholder fetch."""
    return {"key": "value"}
