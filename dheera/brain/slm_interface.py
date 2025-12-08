# brain/slm_interface.py
"""
Dheera SLM Interface
Abstract interface for connecting to various LLM providers.
Supports: Ollama, OpenAI, Anthropic, or any OpenAI-compatible API.
Version 0.2.0 - Enhanced with retry logic, streaming, and better error handling.
"""

import os
import json
import time
import asyncio
import httpx
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Generator, AsyncGenerator
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
    latency_ms: float = 0.0
    
    @property
    def success(self) -> bool:
        return len(self.content) > 0 and not self.content.startswith("Error:")
    
    @property
    def is_error(self) -> bool:
        return self.finish_reason == "error" or self.content.startswith("Error:")


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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {"model": getattr(self, 'model', 'unknown')}


class OllamaSLM(BaseSLM):
    """Ollama local model interface with enhanced features."""
    
    def __init__(
        self, 
        model: str = "phi3:latest",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = httpx.Client(timeout=timeout)
        self._last_health_check: Optional[float] = None
        self._is_healthy: bool = False
        
    def is_available(self) -> bool:
        """Check if Ollama is running and responsive."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
            self._is_healthy = response.status_code == 200
            self._last_health_check = time.time()
            return self._is_healthy
        except Exception:
            self._is_healthy = False
            return False
    
    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            pass
        return []
    
    def is_model_available(self, model_name: Optional[str] = None) -> bool:
        """Check if a specific model is available."""
        model_to_check = model_name or self.model
        available_models = self.list_models()
        
        # Check exact match or prefix match (e.g., "phi3:latest" matches "phi3")
        for m in available_models:
            if m == model_to_check or m.startswith(model_to_check.split(":")[0]):
                return True
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        try:
            response = self.client.post(
                f"{self.base_url}/api/show",
                json={"name": self.model}
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return {"model": self.model, "error": "Could not fetch model info"}
    
    def _generate_with_retry(self, payload: Dict, endpoint: str) -> Dict:
        """Execute request with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.post(
                    f"{self.base_url}{endpoint}",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
                
            except httpx.TimeoutException as e:
                last_error = f"Timeout after {self.timeout}s"
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                # Don't retry on 4xx errors (client errors)
                if 400 <= e.response.status_code < 500:
                    break
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        raise Exception(last_error or "Unknown error")
    
    def generate(self, messages: List[Message], **kwargs) -> SLMResponse:
        """Generate response using Ollama chat API."""
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
        
        start_time = time.time()
        
        try:
            data = self._generate_with_retry(payload, "/api/chat")
            latency_ms = (time.time() - start_time) * 1000
            
            return SLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=self.model,
                provider="ollama",
                tokens_used=data.get("eval_count", 0),
                finish_reason=data.get("done_reason", "stop"),
                raw_response=data,
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return SLMResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                provider="ollama",
                finish_reason="error",
                latency_ms=latency_ms,
            )
    
    def generate_stream(self, messages: List[Message], **kwargs) -> Generator[str, None, None]:
        """Generate response with streaming."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def generate_simple(self, prompt: str, **kwargs) -> SLMResponse:
        """Simple generation using /api/generate endpoint (legacy)."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        start_time = time.time()
        
        try:
            data = self._generate_with_retry(payload, "/api/generate")
            latency_ms = (time.time() - start_time) * 1000
            
            return SLMResponse(
                content=data.get("response", ""),
                model=self.model,
                provider="ollama",
                tokens_used=data.get("eval_count", 0),
                finish_reason="stop" if data.get("done", False) else "incomplete",
                raw_response=data,
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return SLMResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                provider="ollama",
                finish_reason="error",
                latency_ms=latency_ms,
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
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = httpx.Client(timeout=timeout)
        
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
        
        start_time = time.time()
        
        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.time() - start_time) * 1000
            
            choice = data.get("choices", [{}])[0]
            usage = data.get("usage", {})
            
            return SLMResponse(
                content=choice.get("message", {}).get("content", ""),
                model=self.model,
                provider="openai",
                tokens_used=usage.get("total_tokens", 0),
                finish_reason=choice.get("finish_reason", "stop"),
                raw_response=data,
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return SLMResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                provider="openai",
                finish_reason="error",
                latency_ms=latency_ms,
            )


class AnthropicSLM(BaseSLM):
    """Anthropic Claude API interface."""
    
    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        
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
        
        start_time = time.time()
            
        try:
            response = self.client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.time() - start_time) * 1000
            
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
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return SLMResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                provider="anthropic",
                finish_reason="error",
                latency_ms=latency_ms,
            )


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
            content=f"[Echo Mode] Received: {last_msg[:200]}...",
            model="echo",
            provider="echo",
            finish_reason="stop",
        )


class SLMInterface:
    """
    Unified SLM interface for Dheera.
    Automatically selects and manages the appropriate provider.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.provider: Optional[BaseSLM] = None
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_tokens": 0,
            "total_latency_ms": 0.0,
        }
        self._initialize_provider()
        
    def _initialize_provider(self):
        """Initialize the SLM provider based on config."""
        provider_name = self.config.get("provider", "ollama")
        model = self.config.get("model", "phi3:latest")
        base_url = self.config.get("base_url", "http://localhost:11434")
        temperature = self.config.get("temperature", 0.7)
        max_tokens = self.config.get("max_tokens", 1024)
        timeout = self.config.get("timeout", 120.0)
        
        if provider_name == "ollama":
            self.provider = OllamaSLM(
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        elif provider_name == "openai":
            self.provider = OpenAISLM(
                model=model,
                api_key=self.config.get("api_key"),
                base_url=self.config.get("base_url", "https://api.openai.com/v1"),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        elif provider_name == "anthropic":
            self.provider = AnthropicSLM(
                model=model,
                api_key=self.config.get("api_key"),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        elif provider_name == "echo":
            self.provider = EchoSLM()
        else:
            # Default to Ollama
            self.provider = OllamaSLM(model=model, base_url=base_url)
    
    def is_available(self) -> bool:
        """Check if the SLM is available."""
        return self.provider is not None and self.provider.is_available()
    
    def get_provider_name(self) -> str:
        """Get the current provider name."""
        if isinstance(self.provider, OllamaSLM):
            return "ollama"
        elif isinstance(self.provider, OpenAISLM):
            return "openai"
        elif isinstance(self.provider, AnthropicSLM):
            return "anthropic"
        elif isinstance(self.provider, EchoSLM):
            return "echo"
        return "unknown"
    
    def get_model_name(self) -> str:
        """Get the current model name."""
        if self.provider:
            return getattr(self.provider, 'model', 'unknown')
        return "unknown"
    
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
        
        # Track stats
        self._stats["total_calls"] += 1
        
        response = self.provider.generate(msg_objects, **kwargs)
        
        # Update stats
        if response.success:
            self._stats["successful_calls"] += 1
            self._stats["total_tokens"] += response.tokens_used
        else:
            self._stats["failed_calls"] += 1
        
        self._stats["total_latency_ms"] += response.latency_ms
        
        return response
    
    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response from the SLM.
        Only supported by Ollama provider.
        
        Args:
            messages: List of {"role": str, "content": str}
            **kwargs: Additional generation parameters
            
        Yields:
            Response content chunks
        """
        if not self.provider:
            yield "Error: No SLM provider configured"
            return
        
        if not isinstance(self.provider, OllamaSLM):
            # Fall back to non-streaming for other providers
            response = self.generate(messages, **kwargs)
            yield response.content
            return
        
        msg_objects = [Message(role=m["role"], content=m["content"]) for m in messages]
        
        for chunk in self.provider.generate_stream(msg_objects, **kwargs):
            yield chunk
    
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
    
    def chat_stream(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Streaming chat convenience method.
        
        Args:
            user_message: The user's message
            system_prompt: Optional system prompt
            history: Optional conversation history
            **kwargs: Additional generation parameters
            
        Yields:
            Response content chunks
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        if history:
            messages.extend(history)
            
        messages.append({"role": "user", "content": user_message})
        
        for chunk in self.generate_stream(messages, **kwargs):
            yield chunk
    
    def use_echo_mode(self):
        """Switch to echo mode (for testing without a model)."""
        self.provider = EchoSLM()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        avg_latency = 0.0
        if self._stats["total_calls"] > 0:
            avg_latency = self._stats["total_latency_ms"] / self._stats["total_calls"]
        
        return {
            **self._stats,
            "avg_latency_ms": round(avg_latency, 2),
            "success_rate": round(
                self._stats["successful_calls"] / max(1, self._stats["total_calls"]) * 100, 2
            ),
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_tokens": 0,
            "total_latency_ms": 0.0,
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the SLM."""
        result = {
            "available": False,
            "provider": self.get_provider_name(),
            "model": self.get_model_name(),
            "error": None,
        }
        
        try:
            result["available"] = self.is_available()
            
            if result["available"] and isinstance(self.provider, OllamaSLM):
                result["model_available"] = self.provider.is_model_available()
                result["available_models"] = self.provider.list_models()[:5]  # First 5
                
        except Exception as e:
            result["error"] = str(e)
        
        return result


# ==================== Quick Test ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing SLM Interface")
    print("=" * 60)
    
    # Test with Ollama
    print("\n1. Testing Ollama connection...")
    slm = SLMInterface({
        "provider": "ollama",
        "model": "phi3:latest",
        "temperature": 0.7,
        "max_tokens": 256,
    })
    
    # Health check
    health = slm.health_check()
    print(f"   Provider: {health['provider']}")
    print(f"   Model: {health['model']}")
    print(f"   Available: {health['available']}")
    
    if health.get('available_models'):
        print(f"   Models: {', '.join(health['available_models'])}")
    
    if slm.is_available():
        print("\n2. Testing chat...")
        response = slm.chat(
            user_message="What is 2+2? Reply in one word.",
            system_prompt="You are a helpful assistant. Be very concise."
        )
        print(f"   Response: {response.content}")
        print(f"   Tokens: {response.tokens_used}")
        print(f"   Latency: {response.latency_ms:.0f}ms")
        print(f"   Success: {response.success}")
        
        print("\n3. Testing streaming...")
        print("   Streaming: ", end="", flush=True)
        for chunk in slm.chat_stream(
            user_message="Count from 1 to 5.",
            system_prompt="Be concise."
        ):
            print(chunk, end="", flush=True)
        print()
        
        print("\n4. Statistics:")
        stats = slm.get_stats()
        print(f"   Total calls: {stats['total_calls']}")
        print(f"   Success rate: {stats['success_rate']}%")
        print(f"   Avg latency: {stats['avg_latency_ms']}ms")
        
    else:
        print("\n⚠️  Ollama not available. Testing echo mode...")
        slm.use_echo_mode()
        response = slm.chat("Hello Dheera!")
        print(f"   Response: {response.content}")
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("=" * 60)
