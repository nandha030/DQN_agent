"""
Python Code Executor Tool for Dheera
Safely executes Python code in a sandboxed environment.
"""

import sys
import io
import traceback
from typing import Dict, Any
from contextlib import redirect_stdout, redirect_stderr


class PythonExecutor:
    """Execute Python code safely."""
    
    FORBIDDEN = ['os.system', 'subprocess', 'eval', 'exec', '__import__', 
                 'open(', 'rm ', 'sudo', 'chmod', 'chown']
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
    
    def is_safe(self, code: str) -> tuple[bool, str]:
        """Check if code is safe to execute."""
        for forbidden in self.FORBIDDEN:
            if forbidden in code:
                return False, f"Forbidden operation: {forbidden}"
        return True, "OK"
    
    def execute(self, code: str) -> Dict[str, Any]:
        """Execute Python code and return results."""
        
        # Safety check
        is_safe, message = self.is_safe(code)
        if not is_safe:
            return {
                "success": False,
                "output": "",
                "error": message,
            }
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Limited globals for safety
        safe_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "True": True,
                "False": False,
                "None": None,
            },
            "math": __import__("math"),
        }
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, safe_globals)
            
            return {
                "success": True,
                "output": stdout_capture.getvalue(),
                "error": stderr_capture.getvalue(),
            }
        except Exception as e:
            return {
                "success": False,
                "output": stdout_capture.getvalue(),
                "error": f"{type(e).__name__}: {str(e)}",
            }


# Register with Dheera
def register_python_tool(tool_registry):
    """Register the Python executor with Dheera's tool registry."""
    executor = PythonExecutor()
    
    def run_python(code: str) -> str:
        result = executor.execute(code)
        if result["success"]:
            return f"Output:\n{result['output']}" if result['output'] else "Code executed successfully (no output)"
        else:
            return f"Error: {result['error']}"
    
    tool_registry.register(
        name="run_python",
        function=run_python,
        description="Execute Python code and return the output",
    )
    
    return executor
