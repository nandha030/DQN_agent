# connectors/tools/web_search.py
"""
Web Search Tool for Dheera
Enables fact-checking and real-time information retrieval
"""

import asyncio
import re
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from duckduckgo_search import DDGS
import httpx


class SearchCache:
    """Simple in-memory cache for search results"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, dict] = {}
        self.ttl = timedelta(seconds=ttl_seconds)
    
    def get(self, query: str) -> Optional[dict]:
        key = query.lower().strip()
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry['timestamp'] < self.ttl:
                return entry['data']
            del self.cache[key]
        return None
    
    def set(self, query: str, data: dict):
        key = query.lower().strip()
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def clear(self):
        self.cache.clear()


class WebSearchTool:
    """Web search capability for Dheera"""
    
    def __init__(self, max_results: int = 5, timeout: int = 10, cache_ttl: int = 300):
        self.max_results = max_results
        self.timeout = timeout
        self.cache = SearchCache(cache_ttl)
        self.search_history: List[dict] = []
        self.total_searches = 0
        
    def search(self, query: str, max_results: Optional[int] = None, use_cache: bool = True) -> dict:
        """
        Search the web using DuckDuckGo
        
        Args:
            query: Search query string
            max_results: Override default max results
            use_cache: Whether to use cached results
            
        Returns:
            dict with results and metadata
        """
        # Check cache first
        if use_cache:
            cached = self.cache.get(query)
            if cached:
                cached['from_cache'] = True
                return cached
        
        results_limit = max_results or self.max_results
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query, 
                    max_results=results_limit,
                    safesearch='moderate'
                ))
            
            # Format results
            formatted = []
            for r in results:
                formatted.append({
                    'title': r.get('title', ''),
                    'url': r.get('href', ''),
                    'snippet': r.get('body', ''),
                })
            
            search_record = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'result_count': len(formatted),
                'results': formatted,
                'success': True,
                'from_cache': False
            }
            
            # Cache the results
            self.cache.set(query, search_record)
            
            # Track history
            self.search_history.append({
                'query': query,
                'timestamp': search_record['timestamp'],
                'result_count': len(formatted)
            })
            self.total_searches += 1
            
            return search_record
            
        except Exception as e:
            return {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e),
                'results': [],
                'from_cache': False
            }
    
    def search_news(self, query: str, max_results: Optional[int] = None) -> dict:
        """Search recent news"""
        results_limit = max_results or self.max_results
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.news(
                    query,
                    max_results=results_limit
                ))
            
            formatted = []
            for r in results:
                formatted.append({
                    'title': r.get('title', ''),
                    'url': r.get('url', ''),
                    'snippet': r.get('body', ''),
                    'source': r.get('source', ''),
                    'date': r.get('date', '')
                })
            
            return {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'result_count': len(formatted),
                'results': formatted,
                'success': True,
                'type': 'news'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'results': [], 'type': 'news'}
    
    def fact_check(self, claim: str) -> dict:
        """
        Attempt to verify a claim by searching for evidence
        
        Returns search results with relevance scoring
        """
        # Search for the claim + fact check keywords
        queries = [
            claim,
            f"{claim} fact check",
            f"is it true {claim}"
        ]
        
        all_results = []
        for q in queries:
            result = self.search(q, max_results=3, use_cache=False)
            if result['success']:
                all_results.extend(result['results'])
        
        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r['url'] not in seen_urls:
                seen_urls.add(r['url'])
                unique_results.append(r)
        
        return {
            'claim': claim,
            'timestamp': datetime.now().isoformat(),
            'evidence_count': len(unique_results),
            'evidence': unique_results[:self.max_results],
            'success': True
        }
    
    def get_page_content(self, url: str, max_length: int = 4000) -> dict:
        """Fetch and extract text content from a URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, headers=headers, follow_redirects=True)
                response.raise_for_status()
                
            content = response.text
            
            # HTML tag removal
            text = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return {
                'url': url,
                'content': text[:max_length],
                'truncated': len(text) > max_length,
                'full_length': len(text),
                'success': True
            }
            
        except Exception as e:
            return {'url': url, 'success': False, 'error': str(e)}
    
    def get_stats(self) -> dict:
        """Get search statistics"""
        return {
            'total_searches': self.total_searches,
            'cache_size': len(self.cache.cache),
            'recent_queries': [h['query'] for h in self.search_history[-5:]]
        }


class WebSearchAction:
    """
    Action wrapper for DQN integration
    Maps web search to Dheera's action space
    """
    
    ACTION_ID = 3  # SEARCH_WEB action
    
    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.tool = WebSearchTool(
            max_results=config.get('max_results', 5),
            timeout=config.get('timeout', 10),
            cache_ttl=config.get('cache_ttl', 300)
        )
        self.name = "SEARCH_WEB"
        self.description = "Search the internet for current information"
        
    async def execute(self, context: dict) -> dict:
        """
        Execute web search based on conversation context
        
        Args:
            context: Contains 'query' or extracts from 'user_message'
        """
        query = context.get('query') or context.get('user_message', '')
        search_type = context.get('search_type', 'general')
        
        if not query:
            return {
                'success': False,
                'error': 'No search query provided',
                'action': self.name
            }
        
        # Clean query - extract key terms if it's a full question
        clean_query = self._extract_search_terms(query)
        
        # Execute appropriate search type
        if search_type == 'news':
            result = self.tool.search_news(clean_query)
        elif search_type == 'fact_check':
            result = self.tool.fact_check(clean_query)
        else:
            result = self.tool.search(clean_query)
        
        # Format for SLM consumption
        if result['success'] and result.get('results') or result.get('evidence'):
            search_context = self._format_for_slm(result)
            return {
                'success': True,
                'action': self.name,
                'search_results': result.get('results') or result.get('evidence', []),
                'context_for_slm': search_context,
                'result_count': result.get('result_count') or result.get('evidence_count', 0),
                'query_used': clean_query,
                'from_cache': result.get('from_cache', False)
            }
        
        return {
            'success': False,
            'action': self.name,
            'error': result.get('error', 'No results found'),
            'query_used': clean_query
        }
    
    def execute_sync(self, context: dict) -> dict:
        """Synchronous execution wrapper"""
        return asyncio.get_event_loop().run_until_complete(self.execute(context))
    
    def _extract_search_terms(self, text: str) -> str:
        """Extract key search terms from natural language"""
        # Remove common question words
        stop_words = {'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 
                      'the', 'a', 'an', 'can', 'could', 'would', 'should', 'do', 
                      'does', 'did', 'please', 'tell', 'me', 'about', 'explain'}
        
        words = text.lower().split()
        key_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # If we filtered too much, use original
        if len(key_words) < 2:
            return text
        
        return ' '.join(key_words[:8])  # Max 8 key terms
    
    def _format_for_slm(self, result: dict) -> str:
        """Format search results as context for the SLM"""
        results = result.get('results') or result.get('evidence', [])
        query = result.get('query') or result.get('claim', 'search')
        
        lines = [f"[Web Search Results for: {query}]\n"]
        
        for i, r in enumerate(results[:5], 1):
            lines.append(f"{i}. **{r['title']}**")
            lines.append(f"   {r['snippet'][:200]}...")
            lines.append(f"   Source: {r['url']}\n")
        
        lines.append("[Use this information to answer the user's question accurately.]")
        
        return '\n'.join(lines)


# Standalone test
if __name__ == "__main__":
    print("🔍 Testing Dheera Web Search Tool\n")
    
    tool = WebSearchTool()
    
    # Test basic search
    print("Test 1: Basic Search")
    result = tool.search("Python programming latest version")
    if result['success']:
        print(f"  ✓ Found {result['result_count']} results")
        for r in result['results'][:2]:
            print(f"    - {r['title'][:50]}...")
    else:
        print(f"  ✗ Error: {result.get('error')}")
    
    # Test news search
    print("\nTest 2: News Search")
    news_result = tool.search_news("artificial intelligence")
    if news_result['success']:
        print(f"  ✓ Found {news_result['result_count']} news articles")
    else:
        print(f"  ✗ Error: {news_result.get('error')}")
    
    # Test fact check
    print("\nTest 3: Fact Check")
    fact_result = tool.fact_check("Python was created by Guido van Rossum")
    print(f"  ✓ Found {fact_result['evidence_count']} pieces of evidence")
    
    # Test cache
    print("\nTest 4: Cache Test")
    result2 = tool.search("Python programming latest version")
    print(f"  ✓ From cache: {result2.get('from_cache', False)}")
    
    # Stats
    print("\nTest 5: Statistics")
    stats = tool.get_stats()
    print(f"  ✓ Total searches: {stats['total_searches']}")
    print(f"  ✓ Cache size: {stats['cache_size']}")
    
    # Test action wrapper
    print("\nTest 6: Action Wrapper")
    action = WebSearchAction()
    ctx_result = asyncio.run(action.execute({
        'user_message': 'What is the latest version of Ollama?'
    }))
    print(f"  ✓ Action success: {ctx_result['success']}")
    print(f"  ✓ Query used: {ctx_result.get('query_used', 'N/A')}")
    
    print("\n✅ All tests completed!")
