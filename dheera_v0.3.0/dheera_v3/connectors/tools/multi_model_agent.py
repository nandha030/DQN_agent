#!/usr/bin/env python3
"""
Dheera v0.3.1 - Multi-Model Agent
Query multiple AI models simultaneously and compare responses
"""

import requests
import time
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class ModelResponse:
    """Response from a single model"""
    model_name: str
    provider: str
    response_text: str
    latency_ms: float
    tokens_used: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MultiModelResponse:
    """Combined response from multiple models"""
    query: str
    responses: List[ModelResponse]
    total_time_ms: float
    fastest_model: str
    slowest_model: str
    consensus_summary: Optional[str] = None


class MultiModelAgent:
    """
    Agent that queries multiple AI models simultaneously
    Compares responses and finds consensus
    """

    def __init__(self, api_base: str = "http://localhost:8000"):
        self.api_base = api_base
        self.name = "multi_model_agent"
        self.description = "Query multiple AI models and compare responses"

    def query_all(
        self,
        prompt: str,
        models: List[str],
        max_workers: int = 4
    ) -> MultiModelResponse:
        """
        Query multiple models in parallel

        Args:
            prompt: User query
            models: List of model names (e.g., ["groq_llama", "gemini_flash"])
            max_workers: Max parallel requests

        Returns:
            MultiModelResponse with all results
        """
        start_time = time.time()
        responses = []

        # Query all models in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_model = {
                executor.submit(self._query_single, prompt, model): model
                for model in models
            }

            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    response = future.result()
                    responses.append(response)
                except Exception as e:
                    # Add error response
                    responses.append(ModelResponse(
                        model_name=model,
                        provider="unknown",
                        response_text="",
                        latency_ms=0,
                        error=str(e)
                    ))

        total_time = (time.time() - start_time) * 1000

        # Find fastest and slowest
        valid_responses = [r for r in responses if r.error is None]
        if valid_responses:
            fastest = min(valid_responses, key=lambda r: r.latency_ms)
            slowest = max(valid_responses, key=lambda r: r.latency_ms)
        else:
            fastest_name = slowest_name = "none"

        return MultiModelResponse(
            query=prompt,
            responses=responses,
            total_time_ms=total_time,
            fastest_model=fastest.model_name if valid_responses else "none",
            slowest_model=slowest.model_name if valid_responses else "none"
        )

    def _query_single(self, prompt: str, model: str) -> ModelResponse:
        """Query a single model via backend API"""
        start_time = time.time()

        try:
            # Switch to model
            switch_response = requests.post(
                f"{self.api_base}/api/llm/switch",
                json={"name": model},
                timeout=5
            )

            if switch_response.status_code != 200:
                raise Exception(f"Failed to switch to {model}")

            # Send query
            chat_response = requests.post(
                f"{self.api_base}/api/chat",
                json={"message": prompt},
                timeout=60
            )

            if chat_response.status_code != 200:
                raise Exception(f"Chat failed: {chat_response.text}")

            data = chat_response.json()
            latency = (time.time() - start_time) * 1000

            return ModelResponse(
                model_name=model,
                provider=data.get("metadata", {}).get("provider", "unknown"),
                response_text=data.get("response", ""),
                latency_ms=latency,
                tokens_used=data.get("metadata", {}).get("tokens_used", 0),
                metadata=data.get("metadata", {})
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return ModelResponse(
                model_name=model,
                provider="unknown",
                response_text="",
                latency_ms=latency,
                error=str(e)
            )

    def generate_consensus(
        self,
        multi_response: MultiModelResponse,
        using_model: Optional[str] = None
    ) -> str:
        """
        Use one model to analyze all responses and generate consensus

        Args:
            multi_response: Results from query_all()
            using_model: Model to use for summarization (default: fastest)

        Returns:
            Consensus summary text
        """
        # Get valid responses
        valid_responses = [r for r in multi_response.responses if r.error is None]

        if not valid_responses:
            return "No valid responses to summarize."

        # Build prompt for summarization
        summary_prompt = f"I asked multiple AI models: '{multi_response.query}'\n\n"
        summary_prompt += "Here are their responses:\n\n"

        for i, resp in enumerate(valid_responses, 1):
            summary_prompt += f"{i}. {resp.model_name} ({resp.latency_ms:.0f}ms):\n"
            summary_prompt += f"{resp.response_text}\n\n"

        summary_prompt += "Please provide a consensus summary that combines the best parts of all responses. "
        summary_prompt += "Highlight agreements and note any disagreements. Be concise."

        # Use fastest model or specified model for summarization
        if using_model is None:
            using_model = multi_response.fastest_model

        # Query for consensus
        consensus_response = self._query_single(summary_prompt, using_model)

        if consensus_response.error:
            return f"Failed to generate consensus: {consensus_response.error}"

        return consensus_response.response_text

    def format_comparison(self, multi_response: MultiModelResponse) -> str:
        """Format multi-model response as readable comparison"""
        output = []
        output.append(f"🤖 Multi-Model Query: {multi_response.query}")
        output.append(f"⏱️ Total Time: {multi_response.total_time_ms:.0f}ms")
        output.append(f"🏆 Fastest: {multi_response.fastest_model}")
        output.append("")

        for resp in multi_response.responses:
            if resp.error:
                output.append(f"❌ {resp.model_name}: {resp.error}")
            else:
                output.append(f"✅ {resp.model_name} ({resp.latency_ms:.0f}ms)")
                output.append(f"{resp.response_text}")
            output.append("")

        if multi_response.consensus_summary:
            output.append("🎯 Consensus Summary:")
            output.append(multi_response.consensus_summary)

        return "\n".join(output)

    def get_available_models(self) -> List[str]:
        """Get list of available models from backend"""
        try:
            response = requests.get(f"{self.api_base}/api/llm/providers", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [p["name"] for p in data.get("providers", [])]
        except:
            pass
        return []


# ==================== Example Usage ====================
if __name__ == "__main__":
    print("🤖 Testing Multi-Model Agent...")

    agent = MultiModelAgent()

    # Get available models
    print("\n📋 Available models:")
    models = agent.get_available_models()
    for model in models:
        print(f"  - {model}")

    if len(models) < 2:
        print("\n⚠️  Need at least 2 models for comparison")
        print("Add models via GUI: Models → Add API")
        exit()

    # Test multi-model query
    print("\n🔍 Testing multi-model query...")
    test_models = models[:3]  # Use first 3 models
    print(f"Using: {', '.join(test_models)}")

    result = agent.query_all(
        prompt="What is 2+2? Explain briefly.",
        models=test_models
    )

    print(agent.format_comparison(result))

    # Test consensus
    print("\n🎯 Generating consensus...")
    consensus = agent.generate_consensus(result)
    print(f"Consensus: {consensus}")
