"""
Dheera SLM Interface
Abstract interface for connecting to various LLM providers.
Supports: Ollama, OpenAI, Anthropic, or any OpenAI-compatible API.
"""

import os
import json
import httpx
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Generator
from enum import Enum


class SLMProvider(Enum):
    """Supported SLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # Generic OpenAI-compatible


@dataclass
class SLMResponse:
    """Response from SLM."""
    content: str
    model: str
    provider: str
    tokens_used: int = 0
    finish_reason: str = "stop"
    raw_response: Optional[Dict] = None
    
    @property
    def success(self) -> bool:
        return len(self.content) > 0


@dataclass
class Message:
    """Chat message structure."""
    role: str  # "system", "user", "assistant"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class BaseSLM(ABC):
    """Abstract base class for SLM providers."""
    
    @abstractmethod
    def generate(self, messages: List[Message], **kwargs) -> SLMResponse:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass


class OllamaSLM(BaseSLM):
    """Ollama local model interface."""
    
    def __init__(
        self, 
        model: str = "phi3:mini",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = httpx.Client(timeout=120.0)
        
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    def generate(self, messages: List[Message], **kwargs) -> SLMResponse:
        """Generate response using Ollama."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        try:
            response = self.client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            return SLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=self.model,
                provider="ollama",
                tokens_used=data.get("eval_count", 0),
                finish_reason=data.get("done_reason", "stop"),
                raw_response=data,
            )
        except Exception as e:
            return SLMResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                provider="ollama",
                finish_reason="error",
            )


class OpenAISLM(BaseSLM):
    """OpenAI API interface."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = httpx.Client(timeout=60.0)
        
    def is_available(self) -> bool:
        """Check if API key is set."""
        return self.api_key is not None and len(self.api_key) > 0
    
    def generate(self, messages: List[Message], **kwargs) -> SLMResponse:
        """Generate response using OpenAI API."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            choice = data.get("choices", [{}])[0]
            usage = data.get("usage", {})
            
            return SLMResponse(
                content=choice.get("message", {}).get("content", ""),
                model=self.model,
                provider="openai",
                tokens_used=usage.get("total_tokens", 0),
                finish_reason=choice.get("finish_reason", "stop"),
                raw_response=data,
            )
        except Exception as e:
            return SLMResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                provider="openai",
                finish_reason="error",
            )


class AnthropicSLM(BaseSLM):
    """Anthropic Claude API interface."""
    
    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = httpx.Client(timeout=60.0)
        
    def is_available(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0
    
    def generate(self, messages: List[Message], **kwargs) -> SLMResponse:
        """Generate response using Anthropic API."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Extract system message
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                chat_messages.append(m.to_dict())
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": chat_messages,
        }
        
        if system_msg:
            payload["system"] = system_msg
            
        try:
            response = self.client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            content = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    content += block.get("text", "")
            
            usage = data.get("usage", {})
            
            return SLMResponse(
                content=content,
                model=self.model,
                provider="anthropic",
                tokens_used=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                finish_reason=data.get("stop_reason", "stop"),
                raw_response=data,
            )
        except Exception as e:
            return SLMResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                provider="anthropic",
                finish_reason="error",
            )


class SLMInterface:
    """
    Unified SLM interface for Dheera.
    Automatically selects and manages the appropriate provider.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.provider: Optional[BaseSLM] = None
        self._initialize_provider()
        
    def _initialize_provider(self):
        """Initialize the SLM provider based on config."""
        provider_name = self.config.get("provider", "ollama")
        model = self.config.get("model", "phi3:mini")
        base_url = self.config.get("base_url", "http://localhost:11434")
        temperature = self.config.get("temperature", 0.7)
        max_tokens = self.config.get("max_tokens", 1024)
        
        if provider_name == "ollama":
            self.provider = OllamaSLM(
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif provider_name == "openai":
            self.provider = OpenAISLM(
                model=model,
                api_key=self.config.get("api_key"),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif provider_name == "anthropic":
            self.provider = AnthropicSLM(
                model=model,
                api_key=self.config.get("api_key"),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            # Default to Ollama
            self.provider = OllamaSLM(model=model)
    
    def is_available(self) -> bool:
        """Check if the SLM is available."""
        return self.provider is not None and self.provider.is_available()
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> SLMResponse:
        """
        Generate a response from the SLM.
        
        Args:
            messages: List of {"role": str, "content": str}
            **kwargs: Additional generation parameters
            
        Returns:
            SLMResponse with generated content
        """
        if not self.provider:
            return SLMResponse(
                content="Error: No SLM provider configured",
                model="none",
                provider="none",
                finish_reason="error",
            )
        
        # Convert to Message objects
        msg_objects = [Message(role=m["role"], content=m["content"]) for m in messages]
        return self.provider.generate(msg_objects, **kwargs)
    
    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> SLMResponse:
        """
        Convenience method for single-turn chat.
        
        Args:
            user_message: The user's message
            system_prompt: Optional system prompt
            history: Optional conversation history
            **kwargs: Additional generation parameters
            
        Returns:
            SLMResponse
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        if history:
            messages.extend(history)
            
        messages.append({"role": "user", "content": user_message})
        
        return self.generate(messages, **kwargs)


# Fallback for when no SLM is available
class EchoSLM(BaseSLM):
    """Fallback SLM that echoes input (for testing without a model)."""
    
    def __init__(self):
        self.model = "echo"
        
    def is_available(self) -> bool:
        return True
    
    def generate(self, messages: List[Message], **kwargs) -> SLMResponse:
        last_msg = messages[-1].content if messages else "No input"
        return SLMResponse(
            content=f"[Echo Mode] Received: {last_msg[:100]}...",
            model="echo",
            provider="echo",
            finish_reason="stop",
        )


# Quick test
if __name__ == "__main__":
    # Test with Ollama (if available)
    slm = SLMInterface({"provider": "ollama", "model": "phi3:mini"})
    
    if slm.is_available():
        print("SLM is available!")
        response = slm.chat(
            user_message="What is 2+2?",
            system_prompt="You are a helpful assistant. Be concise."
        )
        print(f"Response: {response.content}")
    else:
        print("SLM not available. Using echo mode for testing.")
        slm.provider = EchoSLM()
        response = slm.chat("Hello Dheera!")
        print(f"Response: {response.content}")
