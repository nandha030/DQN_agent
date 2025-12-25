# connectors/__init__.py
"""
Dheera v0.3.0 - Connectors Package
External integrations and tools
"""

from .tool_registry import ToolRegistry, Tool, ToolResult, get_registry
from .web_search import WebSearch, NewsSearch
from .chat_interface import ChatInterface, SimpleChatInterface, ChatMessage, ChatResponse

__all__ = [
    "ToolRegistry",
    "Tool",
    "ToolResult",
    "get_registry",
    "WebSearch",
    "NewsSearch",
    "ChatInterface",
    "SimpleChatInterface",
    "ChatMessage",
    "ChatResponse",
]
