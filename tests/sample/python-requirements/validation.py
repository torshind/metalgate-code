"""Address validation utilities."""

REQUIRED_KEYS = {"street", "city", "zip"}


def validate_address(addr: dict) -> bool:
    """Check address has all required keys."""
    return all(k in addr for k in REQUIRED_KEYS)
