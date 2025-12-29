# connectors/chat_interface.py
"""
Dheera v0.3.0 - Chat Interface
Base interface for chat implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """Represents a chat message"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ChatResponse:
    """Response from chat interface"""
    message: str
    metadata: Optional[Dict[str, Any]] = None
    tokens_used: Optional[int] = None


class ChatInterface(ABC):
    """Abstract base class for chat interfaces"""

    @abstractmethod
    def send_message(self, message: str, context: Optional[List[ChatMessage]] = None) -> ChatResponse:
        """
        Send a message and get a response

        Args:
            message: User message
            context: Optional conversation history

        Returns:
            ChatResponse with the reply
        """
        pass

    @abstractmethod
    def get_history(self) -> List[ChatMessage]:
        """Get conversation history"""
        pass

    @abstractmethod
    def clear_history(self) -> None:
        """Clear conversation history"""
        pass


class SimpleChatInterface(ChatInterface):
    """Simple implementation of chat interface"""

    def __init__(self):
        self.history: List[ChatMessage] = []

    def send_message(self, message: str, context: Optional[List[ChatMessage]] = None) -> ChatResponse:
        """Send message and get response"""
        # Add user message to history
        user_msg = ChatMessage(role="user", content=message)
        self.history.append(user_msg)

        # Placeholder response
        response_text = f"Echo: {message}"
        assistant_msg = ChatMessage(role="assistant", content=response_text)
        self.history.append(assistant_msg)

        return ChatResponse(message=response_text, tokens_used=len(message.split()))

    def get_history(self) -> List[ChatMessage]:
        """Get conversation history"""
        return self.history.copy()

    def clear_history(self) -> None:
        """Clear conversation history"""
        self.history.clear()
