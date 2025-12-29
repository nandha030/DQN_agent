# connectors/tools/calculator.py
"""
Dheera v0.3.0 - Calculator Tool
Simple calculator tool for mathematical operations
"""

import operator
from typing import Union


class CalculatorTool:
    """Basic calculator tool for arithmetic operations"""

    OPERATIONS = {
        'add': operator.add,
        'subtract': operator.sub,
        'multiply': operator.mul,
        'divide': operator.truediv,
        'power': operator.pow,
        'modulo': operator.mod,
    }

    def __init__(self):
        self.name = "calculator"
        self.description = "Performs basic arithmetic operations"

    def calculate(self, operation: str, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """
        Perform a calculation

        Args:
            operation: One of 'add', 'subtract', 'multiply', 'divide', 'power', 'modulo'
            a: First number
            b: Second number

        Returns:
            Result of the operation
        """
        if operation not in self.OPERATIONS:
            raise ValueError(f"Unknown operation: {operation}. Available: {list(self.OPERATIONS.keys())}")

        if operation == 'divide' and b == 0:
            raise ValueError("Division by zero")

        return self.OPERATIONS[operation](a, b)

    def execute(self, expression: str = None, operation: str = None, a=None, b=None, **kwargs):
        """
        Execute calculator - supports both expression and operation modes

        Args:
            expression: A simple math expression like "2 + 2" (basic support)
            operation: Operation name (add, subtract, etc.)
            a: First number
            b: Second number
        """
        if expression:
            # Basic expression parsing (limited support)
            try:
                # Simple eval for basic math (sanitized)
                allowed_chars = set('0123456789+-*/.() ')
                if not all(c in allowed_chars for c in expression):
                    raise ValueError("Invalid characters in expression")
                return eval(expression, {"__builtins__": {}}, {})
            except Exception as e:
                raise ValueError(f"Failed to evaluate expression: {e}")

        elif operation and a is not None and b is not None:
            return self.calculate(operation, a, b)

        else:
            raise ValueError("Provide either 'expression' or 'operation' with 'a' and 'b'")
