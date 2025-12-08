"""
Dheera Connectors Module
Interfaces for chat, tools, and external systems.
"""

from connectors.chat_interface import ChatInterface, ChatMessage, ChatSession
from connectors.tool_registry import ToolRegistry, Tool, ToolResult

# Web search is now integrated into tool_registry.py
# Import from tools subdirectory if needed separately
try:
    from connectors.tools.web_search import WebSearchTool, WebSearchAction
except ImportError:
    # WebSearchTool is built into ToolRegistry, this import is optional
    WebSearchTool = None
    WebSearchAction = None

__all__ = [
    "ChatInterface",
    "ChatMessage", 
    "ChatSession",
    "ToolRegistry",
    "Tool",
    "ToolResult",
    "WebSearchTool",
    "WebSearchAction",
]
