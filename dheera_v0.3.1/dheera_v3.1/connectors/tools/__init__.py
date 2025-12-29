# tools/__init__.py
"""
Dheera v0.3.0 - Tools Package
Available tools: Calculator, WebSearch, Registry
"""

from ..tool_registry import ToolRegistry, Tool, ToolResult, get_registry
from .calculator import CalculatorTool
from ..web_search import WebSearch, NewsSearch

__all__ = [
    "ToolRegistry",
    "Tool", 
    "ToolResult",
    "get_registry",
    "CalculatorTool",
    "WebSearch",
    "NewsSearch",
]
