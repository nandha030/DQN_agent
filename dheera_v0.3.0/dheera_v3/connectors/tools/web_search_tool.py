#!/usr/bin/env python3
"""
Dheera v0.3.1 - Web Search Tool
Search the internet like ChatGPT and Claude
Supports: DuckDuckGo (free), SerpAPI, Brave Search
"""

import requests
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class SearchResult:
    """Single search result"""
    title: str
    url: str
    snippet: str
    source: str = "unknown"


@dataclass
class WebSearchResponse:
    """Response from web search"""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float
    provider: str
    error: Optional[str] = None


class WebSearchTool:
    """
    Tool for searching the web
    Supports multiple providers with automatic fallback
    """

    def __init__(self):
        self.name = "web_search"
        self.description = "Search the internet for current information"

        # API keys (optional - will use free DuckDuckGo if not provided)
        self.serpapi_key: Optional[str] = None
        self.brave_key: Optional[str] = None

    def search(
        self,
        query: str,
        num_results: int = 5,
        provider: str = "auto"
    ) -> WebSearchResponse:
        """
        Search the web

        Args:
            query: Search query
            num_results: Number of results to return
            provider: "duckduckgo", "serpapi", "brave", or "auto"

        Returns:
            WebSearchResponse with results
        """
        start_time = time.time()

        # Auto provider selection
        if provider == "auto":
            if self.serpapi_key:
                provider = "serpapi"
            elif self.brave_key:
                provider = "brave"
            else:
                provider = "duckduckgo"

        # Try selected provider
        try:
            if provider == "duckduckgo":
                results = self._search_duckduckgo(query, num_results)
            elif provider == "serpapi":
                results = self._search_serpapi(query, num_results)
            elif provider == "brave":
                results = self._search_brave(query, num_results)
            else:
                # Fallback to DuckDuckGo
                results = self._search_duckduckgo(query, num_results)

            search_time = (time.time() - start_time) * 1000

            return WebSearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                search_time_ms=search_time,
                provider=provider,
            )

        except Exception as e:
            # Fallback to DuckDuckGo if primary fails
            if provider != "duckduckgo":
                try:
                    results = self._search_duckduckgo(query, num_results)
                    search_time = (time.time() - start_time) * 1000

                    return WebSearchResponse(
                        query=query,
                        results=results,
                        total_results=len(results),
                        search_time_ms=search_time,
                        provider="duckduckgo (fallback)",
                    )
                except:
                    pass

            # Complete failure
            return WebSearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time_ms=(time.time() - start_time) * 1000,
                provider=provider,
                error=str(e),
            )

    def _search_duckduckgo(self, query: str, num_results: int) -> List[SearchResult]:
        """
        Search using DuckDuckGo (FREE - no API key needed)
        Uses HTML scraping since official API only returns instant answers
        """
        results = []

        try:
            # Try DuckDuckGo instant answer API first
            url = f"https://api.duckduckgo.com/?q={requests.utils.quote(query)}&format=json"
            response = requests.get(url, timeout=5)
            data = response.json()

            # Get instant answer if available
            if data.get("AbstractText"):
                results.append(SearchResult(
                    title=data.get("Heading", query),
                    url=data.get("AbstractURL", ""),
                    snippet=data.get("AbstractText", ""),
                    source="DuckDuckGo Instant Answer"
                ))

            # Get related topics
            for topic in data.get("RelatedTopics", [])[:num_results - len(results)]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append(SearchResult(
                        title=topic.get("Text", "")[:100],
                        url=topic.get("FirstURL", ""),
                        snippet=topic.get("Text", ""),
                        source="DuckDuckGo"
                    ))

            # If we don't have enough results, use alternative method
            if len(results) < num_results:
                # Use DuckDuckGo HTML (simple parsing)
                html_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
                response = requests.get(html_url, timeout=5, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; DheeraBot/1.0)'
                })

                # Very basic HTML parsing (you can improve this)
                html = response.text
                import re

                # Extract result snippets (basic regex)
                links = re.findall(r'<a class="result__a"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>', html)
                snippets = re.findall(r'<a class="result__snippet"[^>]*>([^<]+)</a>', html)

                for i, ((url, title), snippet) in enumerate(zip(links[:num_results], snippets[:num_results])):
                    if len(results) >= num_results:
                        break

                    results.append(SearchResult(
                        title=title.strip(),
                        url=url,
                        snippet=snippet.strip(),
                        source="DuckDuckGo"
                    ))

        except Exception as e:
            print(f"DuckDuckGo search error: {e}")

        return results[:num_results]

    def _search_serpapi(self, query: str, num_results: int) -> List[SearchResult]:
        """Search using SerpAPI (requires API key)"""
        if not self.serpapi_key:
            raise ValueError("SerpAPI key not configured")

        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": self.serpapi_key,
            "num": num_results,
            "engine": "google",  # or "bing", "duckduckgo", etc.
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        results = []
        for item in data.get("organic_results", [])[:num_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source="Google (SerpAPI)"
            ))

        return results

    def _search_brave(self, query: str, num_results: int) -> List[SearchResult]:
        """Search using Brave Search API (requires API key)"""
        if not self.brave_key:
            raise ValueError("Brave Search API key not configured")

        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_key,
        }
        params = {
            "q": query,
            "count": num_results,
        }

        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()

        results = []
        for item in data.get("web", {}).get("results", [])[:num_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                source="Brave Search"
            ))

        return results

    def format_results(self, response: WebSearchResponse) -> str:
        """Format search results as readable text"""
        if response.error:
            return f"Search failed: {response.error}"

        if not response.results:
            return f"No results found for: {response.query}"

        output = []
        output.append(f"🔍 Search results for: {response.query}")
        output.append(f"Provider: {response.provider} | Time: {response.search_time_ms:.0f}ms | Results: {response.total_results}")
        output.append("")

        for i, result in enumerate(response.results, 1):
            output.append(f"{i}. **{result.title}**")
            output.append(f"   {result.snippet}")
            output.append(f"   🔗 {result.url}")
            output.append("")

        return "\n".join(output)

    def format_for_llm(self, response: WebSearchResponse) -> str:
        """Format search results for LLM context"""
        if response.error or not response.results:
            return f"Search for '{response.query}' returned no results."

        context = f"Web search results for: {response.query}\n\n"

        for i, result in enumerate(response.results, 1):
            context += f"[{i}] {result.title}\n"
            context += f"{result.snippet}\n"
            context += f"Source: {result.url}\n\n"

        return context


# ==================== Example Usage ====================
if __name__ == "__main__":
    print("🔍 Testing Web Search Tool...")

    tool = WebSearchTool()

    # Test DuckDuckGo (free)
    print("\n1. Testing DuckDuckGo (free)...")
    response = tool.search("Python programming language", num_results=3)
    print(tool.format_results(response))

    # Test with different query
    print("\n2. Testing current events...")
    response = tool.search("latest AI news 2025", num_results=5)
    print(tool.format_results(response))

    # Show LLM-formatted output
    print("\n3. LLM-formatted output:")
    print(tool.format_for_llm(response))
