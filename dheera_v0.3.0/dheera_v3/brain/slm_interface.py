# brain/slm_interface.py
"""
Dheera v0.3.1 - SLM Interface (Updated)
Interface to Small Language Models (Ollama, OpenAI, Anthropic).
Default: Ollama with Phi3:mini

Key upgrades:
- SLMResponse now has an explicit `error` field (first-class)
- Ollama errors do NOT return "Error: ..." inside text (prevents executor treating it as content)
- Timeout is detected and labeled finish_reason="timeout"
- Health check uses finish_reason, not substring "error" in text
- Stats remain backward compatible
"""

import json
import time
from typing import Optional, Dict, Any, Generator, List
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class SLMResponse:
    """Response from SLM."""
    text: str
    model: str
    tokens_used: int
    latency_ms: float
    finish_reason: str = "complete"  # complete | error | timeout
    metadata: Dict[str, Any] = None
    error: Optional[str] = None      # NEW: first-class error field

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SLMProvider(ABC):
    """Abstract base class for SLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> SLMResponse:
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError


class OllamaProvider(SLMProvider):
    """Ollama provider for local models."""

    def __init__(
        self,
        model: str = "phi3:mini",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._available = None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> SLMResponse:
        import requests

        start_time = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            text = (data.get("message", {}) or {}).get("content", "") or ""
            tokens = data.get("eval_count")
            if tokens is None:
                # Rough fallback if Ollama doesn't return eval_count
                tokens = max(1, len(text.split())) if text else 0

            return SLMResponse(
                text=text,
                model=self.model,
                tokens_used=int(tokens),
                latency_ms=(time.time() - start_time) * 1000,
                finish_reason="complete",
                metadata={
                    "provider": "ollama",
                    "total_duration": data.get("total_duration"),
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "eval_count": data.get("eval_count"),
                },
                error=None,
            )

        except requests.exceptions.Timeout as e:
            return SLMResponse(
                text="",
                model=self.model,
                tokens_used=0,
                latency_ms=(time.time() - start_time) * 1000,
                finish_reason="timeout",
                metadata={"provider": "ollama"},
                error=f"TIMEOUT: {str(e)}",
            )

        except Exception as e:
            return SLMResponse(
                text="",
                model=self.model,
                tokens_used=0,
                latency_ms=(time.time() - start_time) * 1000,
                finish_reason="error",
                metadata={"provider": "ollama"},
                error=str(e),
            )

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        Stream response tokens.
        Note: streaming returns chunks, so errors are yielded as a final message.
        """
        import requests

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                content = (data.get("message", {}) or {}).get("content", "")
                if content:
                    yield content
                if data.get("done"):
                    break

        except requests.exceptions.Timeout as e:
            yield f"[TIMEOUT] {str(e)}"
        except Exception as e:
            yield f"[ERROR] {str(e)}"

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        if self._available is not None:
            return self._available

        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            self._available = response.status_code == 200
        except Exception:
            self._available = False

        return self._available


class EchoProvider(SLMProvider):
    """Echo provider for testing (no actual LLM)."""

    def __init__(self, model: str = "echo"):
        self.model = model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> SLMResponse:
        # Echo should NEVER mark as error
        text = f"[Echo] You said: {prompt[:200]}"
        return SLMResponse(
            text=text,
            model=self.model,
            tokens_used=len(text.split()),
            latency_ms=1.0,
            finish_reason="complete",
            metadata={"provider": "echo"},
            error=None,
        )

    def is_available(self) -> bool:
        return True


class SLMInterface:
    """
    Main interface for SLM interactions.
    Supports multiple providers with automatic fallback.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "phi3:mini",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        default_temperature: float = 0.7,
        default_max_tokens: int = 512,
    ):
        self.provider_name = provider
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

        if provider == "ollama":
            self.provider = OllamaProvider(
                model=model,
                base_url=base_url,
                timeout=timeout,
            )
        elif provider == "echo":
            self.provider = EchoProvider(model=model)
        else:
            print(f"⚠ Unknown provider '{provider}', using echo")
            self.provider = EchoProvider()

        # Statistics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
        self.error_count = 0
        self.timeout_count = 0

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> SLMResponse:
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature or self.default_temperature

        self.total_requests += 1

        response = self.provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        # Track stats safely
        self.total_tokens += int(getattr(response, "tokens_used", 0) or 0)
        self.total_latency_ms += float(getattr(response, "latency_ms", 0.0) or 0.0)

        if response.finish_reason == "error":
            self.error_count += 1
        elif response.finish_reason == "timeout":
            self.timeout_count += 1

        return response

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> SLMResponse:
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"User: {content}")
            else:
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n".join(prompt_parts)
        if not prompt.endswith("Assistant:"):
            prompt += "\nAssistant:"

        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def is_available(self) -> bool:
        return self.provider.is_available()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self.model,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.total_latency_ms / max(self.total_requests, 1),
            "error_count": self.error_count,
            "timeout_count": self.timeout_count,
            "error_rate": self.error_count / max(self.total_requests, 1),
            "timeout_rate": self.timeout_count / max(self.total_requests, 1),
            "available": self.is_available(),
        }

    def health_check(self) -> Dict[str, Any]:
        start = time.time()
        try:
            response = self.generate(
                prompt="Say 'OK' if you're working.",
                max_tokens=10,
                temperature=0.0,
            )

            status = "healthy" if response.finish_reason == "complete" and response.text.strip() else "unhealthy"

            return {
                "status": status,
                "latency_ms": (time.time() - start) * 1000,
                "response": response.text[:50],
                "finish_reason": response.finish_reason,
                "error": response.error,
                "provider": self.provider_name,
                "model": self.model,
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "provider": self.provider_name,
                "model": self.model,
            }


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing SLMInterface v0.3.1...")

    slm = SLMInterface(provider="echo", model="test")
    r = slm.generate("Hello")
    print(f"✓ Echo: {r.text} | {r.finish_reason} | err={r.error}")

    print("\nTesting Ollama provider...")
    ollama_slm = SLMInterface(provider="ollama", model="phi3:mini")

    if ollama_slm.is_available():
        r = ollama_slm.generate("Say hello in one word.", max_tokens=16, temperature=0.2)
        print(f"✓ Ollama: {r.text[:80]} | {r.finish_reason} | err={r.error}")
        print(f"  Latency: {r.latency_ms:.0f}ms Tokens: {r.tokens_used}")
    else:
        print("⚠ Ollama not available (run 'ollama serve' to start)")

    print("\n✓ Stats:", slm.get_stats())
    print("✓ Health:", slm.health_check())

    print("\n✅ SLM interface tests passed!")
