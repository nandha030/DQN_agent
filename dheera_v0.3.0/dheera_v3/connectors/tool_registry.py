# connectors/tool_registry.py
"""
Dheera v0.3.0 - Tool Registry
Manages available tools for the agent
"""

from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result from a tool execution"""
    success: bool
    output: Any
    error: Optional[str] = None


@dataclass
class Tool:
    """Tool definition"""
    name: str
    description: str
    execute: Callable
    parameters: Dict[str, Any]


class ToolRegistry:
    """Registry for managing available tools"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a new tool"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names"""
        return list(self._tools.keys())

    def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name"""
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, output=None, error=f"Tool '{name}' not found")

        try:
            output = tool.execute(**kwargs)
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


# Global registry instance
_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the global tool registry"""
    return _registry
