"""
Dheera Tool Registry
Registry for external tools and capabilities that Dheera can invoke.
"""

import time
import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum
import json


class ToolCategory(Enum):
    """Categories of tools."""
    UTILITY = "utility"
    CODE = "code"
    SEARCH = "search"
    FILE = "file"
    API = "api"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "int", "float", "bool", "list", "dict"
    description: str
    required: bool = True
    default: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
            "default": self.default,
        }


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }


@dataclass
class Tool:
    """Definition of a tool that Dheera can use."""
    name: str
    description: str
    function: Callable
    category: ToolCategory = ToolCategory.CUSTOM
    parameters: List[ToolParameter] = field(default_factory=list)
    requires_approval: bool = False
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": [p.to_dict() for p in self.parameters],
            "requires_approval": self.requires_approval,
            "enabled": self.enabled,
        }
    
    def get_signature(self) -> str:
        """Get a human-readable signature."""
        params = ", ".join([
            f"{p.name}: {p.type}" + ("" if p.required else f" = {p.default}")
            for p in self.parameters
        ])
        return f"{self.name}({params})"
    
    def validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """Validate parameters against schema. Returns error message or None."""
        for param in self.parameters:
            if param.required and param.name not in params:
                return f"Missing required parameter: {param.name}"
        return None


class ToolRegistry:
    """
    Registry for managing Dheera's tools.
    
    Features:
    - Register/unregister tools
    - Execute tools with validation
    - Generate tool descriptions for SLM
    - Track tool usage statistics
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        
        # Register built-in tools
        self._register_builtins()
    
    def _register_builtins(self):
        """Register built-in utility tools."""
        
        # Calculator tool
        self.register(
            name="calculator",
            description="Evaluate mathematical expressions",
            function=self._builtin_calculator,
            category=ToolCategory.UTILITY,
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Mathematical expression to evaluate (e.g., '2 + 2 * 3')"
                )
            ]
        )
        
        # Current time tool
        self.register(
            name="get_time",
            description="Get the current date and time",
            function=self._builtin_get_time,
            category=ToolCategory.UTILITY,
            parameters=[]
        )
        
        # JSON formatter tool
        self.register(
            name="format_json",
            description="Format and validate JSON",
            function=self._builtin_format_json,
            category=ToolCategory.UTILITY,
            parameters=[
                ToolParameter(
                    name="json_string",
                    type="string",
                    description="JSON string to format"
                )
            ]
        )
    
    def _builtin_calculator(self, expression: str) -> float:
        """Safe mathematical expression evaluator."""
        # Only allow safe operations
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Invalid characters in expression")
        
        # Evaluate safely
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return float(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate: {e}")
    
    def _builtin_get_time(self) -> str:
        """Get current time."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _builtin_format_json(self, json_string: str) -> str:
        """Format JSON string."""
        try:
            data = json.loads(json_string)
            return json.dumps(data, indent=2)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    
    def register(
        self,
        name: str,
        description: str,
        function: Callable,
        category: ToolCategory = ToolCategory.CUSTOM,
        parameters: Optional[List[ToolParameter]] = None,
        requires_approval: bool = False,
        **metadata
    ) -> Tool:
        """
        Register a new tool.
        
        Args:
            name: Unique tool name
            description: What the tool does
            function: The callable to execute
            category: Tool category
            parameters: Parameter definitions
            requires_approval: Whether human approval is needed
            **metadata: Additional metadata
            
        Returns:
            The registered Tool
        """
        if parameters is None:
            # Auto-extract parameters from function signature
            parameters = self._extract_parameters(function)
        
        tool = Tool(
            name=name,
            description=description,
            function=function,
            category=category,
            parameters=parameters,
            requires_approval=requires_approval,
            metadata=metadata,
        )
        
        self.tools[name] = tool
        self.usage_stats[name] = {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "total_time": 0.0,
        }
        
        return tool
    
    def _extract_parameters(self, func: Callable) -> List[ToolParameter]:
        """Extract parameters from function signature."""
        params = []
        sig = inspect.signature(func)
        
        for name, param in sig.parameters.items():
            if name in ('self', 'cls'):
                continue
            
            # Determine type
            type_str = "string"
            if param.annotation != inspect.Parameter.empty:
                type_map = {
                    str: "string",
                    int: "int",
                    float: "float",
                    bool: "bool",
                    list: "list",
                    dict: "dict",
                }
                type_str = type_map.get(param.annotation, "string")
            
            # Check if required
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default
            
            params.append(ToolParameter(
                name=name,
                type=type_str,
                description=f"Parameter: {name}",
                required=required,
                default=default,
            ))
        
        return params
    
    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self.tools:
            del self.tools[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def execute(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> ToolResult:
        """
        Execute a tool.
        
        Args:
            name: Tool name
            params: Parameters to pass
            timeout: Execution timeout in seconds
            
        Returns:
            ToolResult with output or error
        """
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool not found: {name}"
            )
        
        if not tool.enabled:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool is disabled: {name}"
            )
        
        params = params or {}
        
        # Validate parameters
        validation_error = tool.validate_params(params)
        if validation_error:
            return ToolResult(
                success=False,
                output=None,
                error=validation_error
            )
        
        # Execute
        start_time = time.time()
        try:
            output = tool.function(**params)
            execution_time = time.time() - start_time
            
            # Update stats
            self.usage_stats[name]["calls"] += 1
            self.usage_stats[name]["successes"] += 1
            self.usage_stats[name]["total_time"] += execution_time
            
            return ToolResult(
                success=True,
                output=output,
                execution_time=execution_time,
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Update stats
            self.usage_stats[name]["calls"] += 1
            self.usage_stats[name]["failures"] += 1
            self.usage_stats[name]["total_time"] += execution_time
            
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=execution_time,
            )
    
    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        enabled_only: bool = True,
    ) -> List[Tool]:
        """List available tools."""
        tools = list(self.tools.values())
        
        if category:
            tools = [t for t in tools if t.category == category]
        
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        
        return tools
    
    def get_tools_description(self) -> str:
        """Generate tool descriptions for SLM prompt."""
        lines = ["Available tools:"]
        
        for tool in self.list_tools():
            lines.append(f"\n- {tool.name}: {tool.description}")
            lines.append(f"  Usage: {tool.get_signature()}")
            
            if tool.requires_approval:
                lines.append("  ⚠️ Requires human approval")
        
        return "\n".join(lines)
    
    def get_tool_schema(self) -> List[Dict[str, Any]]:
        """Get tool schemas in OpenAI function-calling format."""
        schemas = []
        
        for tool in self.list_tools():
            schema = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            }
            
            for param in tool.parameters:
                schema["parameters"]["properties"][param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.required:
                    schema["parameters"]["required"].append(param.name)
            
            schemas.append(schema)
        
        return schemas
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all tools."""
        return {
            "total_tools": len(self.tools),
            "enabled_tools": len([t for t in self.tools.values() if t.enabled]),
            "tool_stats": self.usage_stats,
        }


# Quick test
if __name__ == "__main__":
    registry = ToolRegistry()
    
    # Test built-in tools
    print("Testing calculator...")
    result = registry.execute("calculator", {"expression": "2 + 2 * 3"})
    print(f"  2 + 2 * 3 = {result.output}")
    
    print("\nTesting get_time...")
    result = registry.execute("get_time", {})
    print(f"  Current time: {result.output}")
    
    print("\nTesting format_json...")
    result = registry.execute("format_json", {"json_string": '{"name":"Dheera","type":"agent"}'})
    print(f"  Formatted:\n{result.output}")
    
    # Register custom tool
    def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"
    
    registry.register(
        name="greet",
        description="Generate a greeting",
        function=greet,
    )
    
    print("\nTesting custom greet tool...")
    result = registry.execute("greet", {"name": "Nandha"})
    print(f"  {result.output}")
    
    print("\nTool descriptions:")
    print(registry.get_tools_description())
    
    print("\nUsage stats:")
    print(json.dumps(registry.get_stats(), indent=2))
