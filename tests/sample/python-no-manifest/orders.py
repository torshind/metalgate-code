"""Order processing module."""

from validation import validate_address
from utils import format_currency

class Order:
    """An order with address and amount."""
    def __init__(self, address, amount):
        self.address = address
        self.amount = amount

    def process(self):
        """Process the order."""
        if not validate_address(self.address):
            return "invalid"
        formatted = format_currency(self.amount)
        return formatted
