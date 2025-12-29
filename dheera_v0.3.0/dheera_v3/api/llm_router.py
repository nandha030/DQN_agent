#!/usr/bin/env python3
"""
Dheera LLM Router - Hot-swappable LLM backends
Supports: Ollama, OpenAI, Anthropic, LiteLLM, Hugging Face
Inspired by: LiteLLM, OpenUI
"""

import os
import time
from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json


@dataclass
class LLMConfig:
    """Configuration for an LLM provider"""
    provider: str  # "ollama", "openai", "anthropic", "litellm", "huggingface"
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    max_tokens: int = 256
    temperature: float = 0.7
    streaming: bool = False

    # Advanced options
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    # Custom metadata
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LLMResponse:
    """Unified response from any LLM"""
    text: str
    model: str
    provider: str
    tokens_used: int
    latency_ms: float
    finish_reason: str = "complete"
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseLLMProvider(ABC):
    """Abstract base for all LLM providers"""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    def get_info(self) -> Dict[str, Any]:
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "available": self.is_available(),
        }


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider"""

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        import requests

        start = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                f"{self.config.base_url or 'http://localhost:11434'}/api/chat",
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": self.config.max_tokens,
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p or 0.9,
                        "top_k": self.config.top_k or 40,
                    },
                },
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()

            text = data.get("message", {}).get("content", "")
            tokens = data.get("eval_count") or len(text.split())

            return LLMResponse(
                text=text,
                model=self.config.model,
                provider="ollama",
                tokens_used=tokens,
                latency_ms=(time.time() - start) * 1000,
                metadata={
                    "total_duration": data.get("total_duration"),
                    "prompt_eval_count": data.get("prompt_eval_count"),
                },
            )

        except Exception as e:
            return LLMResponse(
                text="",
                model=self.config.model,
                provider="ollama",
                tokens_used=0,
                latency_ms=(time.time() - start) * 1000,
                finish_reason="error",
                error=str(e),
            )

    def is_available(self) -> bool:
        try:
            import requests
            base_url = self.config.base_url or "http://localhost:11434"
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider"""

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError:
            return LLMResponse(
                text="",
                model=self.config.model,
                provider="openai",
                tokens_used=0,
                latency_ms=0,
                finish_reason="error",
                error="openai package not installed. Run: pip install openai",
            )

        start = time.time()

        client = OpenAI(
            api_key=self.config.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
            )

            text = response.choices[0].message.content
            tokens = response.usage.total_tokens

            return LLMResponse(
                text=text,
                model=self.config.model,
                provider="openai",
                tokens_used=tokens,
                latency_ms=(time.time() - start) * 1000,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "finish_reason": response.choices[0].finish_reason,
                },
            )

        except Exception as e:
            return LLMResponse(
                text="",
                model=self.config.model,
                provider="openai",
                tokens_used=0,
                latency_ms=(time.time() - start) * 1000,
                finish_reason="error",
                error=str(e),
            )

    def is_available(self) -> bool:
        return bool(self.config.api_key or os.getenv("OPENAI_API_KEY"))


class AnthropicProvider(BaseLLMProvider):
    """Anthropic (Claude) API provider"""

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        try:
            from anthropic import Anthropic
        except ImportError:
            return LLMResponse(
                text="",
                model=self.config.model,
                provider="anthropic",
                tokens_used=0,
                latency_ms=0,
                finish_reason="error",
                error="anthropic package not installed. Run: pip install anthropic",
            )

        start = time.time()

        client = Anthropic(
            api_key=self.config.api_key or os.getenv("ANTHROPIC_API_KEY"),
            timeout=self.config.timeout,
        )

        try:
            response = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens

            return LLMResponse(
                text=text,
                model=self.config.model,
                provider="anthropic",
                tokens_used=tokens,
                latency_ms=(time.time() - start) * 1000,
                metadata={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "stop_reason": response.stop_reason,
                },
            )

        except Exception as e:
            return LLMResponse(
                text="",
                model=self.config.model,
                provider="anthropic",
                tokens_used=0,
                latency_ms=(time.time() - start) * 1000,
                finish_reason="error",
                error=str(e),
            )

    def is_available(self) -> bool:
        return bool(self.config.api_key or os.getenv("ANTHROPIC_API_KEY"))


class LiteLLMProvider(BaseLLMProvider):
    """LiteLLM unified provider (supports 100+ models)"""

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        try:
            import litellm
        except ImportError:
            return LLMResponse(
                text="",
                model=self.config.model,
                provider="litellm",
                tokens_used=0,
                latency_ms=0,
                finish_reason="error",
                error="litellm package not installed. Run: pip install litellm",
            )

        start = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = litellm.completion(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                api_key=self.config.api_key,
                api_base=self.config.base_url,
                timeout=self.config.timeout,
            )

            text = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else len(text.split())

            return LLMResponse(
                text=text,
                model=self.config.model,
                provider="litellm",
                tokens_used=tokens,
                latency_ms=(time.time() - start) * 1000,
                metadata={
                    "actual_model": response.model,
                    "finish_reason": response.choices[0].finish_reason,
                },
            )

        except Exception as e:
            return LLMResponse(
                text="",
                model=self.config.model,
                provider="litellm",
                tokens_used=0,
                latency_ms=(time.time() - start) * 1000,
                finish_reason="error",
                error=str(e),
            )

    def is_available(self) -> bool:
        try:
            import litellm
            return True
        except:
            return False


class LLMRouter:
    """
    Hot-swappable LLM router
    Manages multiple LLM providers and allows runtime switching
    """

    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.active_provider: Optional[str] = None
        self.provider_classes = {
            "ollama": OllamaProvider,
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "litellm": LiteLLMProvider,
        }

        # Statistics
        self.total_requests = 0
        self.provider_stats: Dict[str, Dict[str, Any]] = {}

    def add_provider(self, name: str, config: LLMConfig) -> bool:
        """Add a new LLM provider"""
        provider_class = self.provider_classes.get(config.provider)
        if not provider_class:
            return False

        self.providers[name] = provider_class(config)
        self.provider_stats[name] = {
            "requests": 0,
            "total_tokens": 0,
            "total_latency_ms": 0,
            "errors": 0,
        }

        # Set as active if first provider
        if self.active_provider is None:
            self.active_provider = name

        return True

    def remove_provider(self, name: str) -> bool:
        """Remove a provider"""
        if name in self.providers:
            del self.providers[name]
            if self.active_provider == name:
                self.active_provider = next(iter(self.providers.keys()), None)
            return True
        return False

    def switch_provider(self, name: str) -> bool:
        """Hot-swap to a different provider"""
        if name in self.providers:
            self.active_provider = name
            return True
        return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate text using active or specified provider

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            provider: Override active provider for this request
        """
        target_provider = provider or self.active_provider

        if not target_provider or target_provider not in self.providers:
            return LLMResponse(
                text="",
                model="",
                provider="none",
                tokens_used=0,
                latency_ms=0,
                finish_reason="error",
                error=f"No provider '{target_provider}' available",
            )

        self.total_requests += 1
        self.provider_stats[target_provider]["requests"] += 1

        response = self.providers[target_provider].generate(prompt, system_prompt)

        # Update stats
        if response.error:
            self.provider_stats[target_provider]["errors"] += 1
        else:
            self.provider_stats[target_provider]["total_tokens"] += response.tokens_used
            self.provider_stats[target_provider]["total_latency_ms"] += response.latency_ms

        return response

    def list_providers(self) -> List[Dict[str, Any]]:
        """List all configured providers"""
        return [
            {
                "name": name,
                "active": name == self.active_provider,
                **provider.get_info(),
                "stats": self.provider_stats.get(name, {}),
            }
            for name, provider in self.providers.items()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        return {
            "total_requests": self.total_requests,
            "active_provider": self.active_provider,
            "providers": self.list_providers(),
        }

    def test_provider(self, name: str) -> Dict[str, Any]:
        """Test a provider with a simple query"""
        if name not in self.providers:
            return {"success": False, "error": f"Provider '{name}' not found"}

        start = time.time()
        response = self.providers[name].generate("Say 'OK' if you're working.", "")
        elapsed = (time.time() - start) * 1000

        return {
            "success": not bool(response.error),
            "available": self.providers[name].is_available(),
            "latency_ms": elapsed,
            "response": response.text[:50],
            "error": response.error,
        }


# ==================== Preset Configurations ====================

PROVIDER_PRESETS = {
    "ollama_phi3": LLMConfig(
        provider="ollama",
        model="phi3:mini",
        base_url="http://localhost:11434",
        timeout=15,
        max_tokens=256,
    ),
    "ollama_gemma": LLMConfig(
        provider="ollama",
        model="gemma:2b",
        base_url="http://localhost:11434",
        timeout=15,
        max_tokens=256,
    ),
    "openai_gpt4": LLMConfig(
        provider="openai",
        model="gpt-4-turbo-preview",
        timeout=30,
        max_tokens=512,
    ),
    "openai_gpt35": LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        timeout=20,
        max_tokens=512,
    ),
    "anthropic_opus": LLMConfig(
        provider="anthropic",
        model="claude-opus-4-20250514",
        timeout=30,
        max_tokens=512,
    ),
    "anthropic_sonnet": LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        timeout=20,
        max_tokens=512,
    ),
    # OpenRouter presets (uses OpenAI-compatible API)
    "openrouter_claude_sonnet": LLMConfig(
        provider="openai",  # Uses OpenAI-compatible API
        model="anthropic/claude-3.5-sonnet",
        base_url="https://openrouter.ai/api/v1",
        timeout=30,
        max_tokens=512,
        metadata={"provider_name": "OpenRouter"},
    ),
    "openrouter_gpt4": LLMConfig(
        provider="openai",
        model="openai/gpt-4-turbo",
        base_url="https://openrouter.ai/api/v1",
        timeout=30,
        max_tokens=512,
        metadata={"provider_name": "OpenRouter"},
    ),
    "openrouter_llama": LLMConfig(
        provider="openai",
        model="meta-llama/llama-3.1-70b-instruct",
        base_url="https://openrouter.ai/api/v1",
        timeout=30,
        max_tokens=512,
        metadata={"provider_name": "OpenRouter"},
    ),
    # Free API providers (100% free - no credit card)
    "groq_llama": LLMConfig(
        provider="openai",  # Groq uses OpenAI-compatible API
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        timeout=20,
        max_tokens=512,
        metadata={"provider_name": "Groq", "free": True, "speed": "ultra-fast"},
    ),
    "groq_mixtral": LLMConfig(
        provider="openai",
        model="mixtral-8x7b-32768",
        base_url="https://api.groq.com/openai/v1",
        timeout=20,
        max_tokens=512,
        metadata={"provider_name": "Groq", "free": True},
    ),
    "gemini_flash": LLMConfig(
        provider="openai",  # Google Gemini via OpenAI-compatible endpoint
        model="gemini-2.0-flash-exp",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        timeout=30,
        max_tokens=512,
        metadata={"provider_name": "Google Gemini", "free": True},
    ),
    "together_llama": LLMConfig(
        provider="openai",  # Together AI uses OpenAI-compatible API
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        base_url="https://api.together.xyz/v1",
        timeout=30,
        max_tokens=512,
        metadata={"provider_name": "Together AI", "free_credits": "$25"},
    ),
    "together_qwen": LLMConfig(
        provider="openai",
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        base_url="https://api.together.xyz/v1",
        timeout=30,
        max_tokens=512,
        metadata={"provider_name": "Together AI", "free_credits": "$25"},
    ),
}


# ==================== Example Usage ====================
if __name__ == "__main__":
    print("🔌 Testing LLM Router...")

    router = LLMRouter()

    # Add Ollama provider
    router.add_provider("local_phi3", PROVIDER_PRESETS["ollama_phi3"])

    # Test it
    test_result = router.test_provider("local_phi3")
    print(f"\n✓ Test result: {test_result}")

    # Generate
    if test_result["available"]:
        response = router.generate("Say hello in one sentence.")
        print(f"\n✓ Response: {response.text}")
        print(f"  Provider: {response.provider}")
        print(f"  Latency: {response.latency_ms:.0f}ms")

    # Stats
    print(f"\n✓ Stats: {router.get_stats()}")

    print("\n✅ LLM Router test complete!")
