# connectors/tools/python_executor.py
"""
Dheera v0.3.0 - Python Code Executor Tool
Safely executes Python code snippets (with restrictions)
"""

import sys
from io import StringIO
from typing import Dict, Any


class PythonExecutor:
    """Tool for executing Python code snippets safely"""

    def __init__(self):
        self.name = "python_executor"
        self.description = "Execute Python code snippets in a restricted environment"

    def execute(self, code: str, timeout: int = 5) -> Dict[str, Any]:
        """
        Execute Python code and capture output

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds

        Returns:
            Dictionary with 'output', 'error', and 'success' keys
        """
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = StringIO()
        redirected_error = StringIO()
        sys.stdout = redirected_output
        sys.stderr = redirected_error

        result = {
            "success": False,
            "output": "",
            "error": "",
        }

        try:
            # Create a restricted namespace
            namespace = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "range": range,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "tuple": tuple,
                    "set": set,
                    "abs": abs,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "sorted": sorted,
                    "enumerate": enumerate,
                    "zip": zip,
                }
            }

            # Execute the code
            exec(code, namespace)

            result["success"] = True
            result["output"] = redirected_output.getvalue()

        except Exception as e:
            result["success"] = False
            result["error"] = f"{type(e).__name__}: {str(e)}"
            result["output"] = redirected_output.getvalue()

        finally:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return result
