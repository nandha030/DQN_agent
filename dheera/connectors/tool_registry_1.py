# connectors/tool_registry.py
"""
Dheera Tool Registry
Registry for external tools and capabilities that Dheera can invoke.
Updated with Web Search integration.
"""

import time
import inspect
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum
import json


class ToolCategory(Enum):
    """Categories of tools."""
    UTILITY = "utility"
    CODE = "code"
    SEARCH = "search"      # Web search, fact-checking
    FILE = "file"
    API = "api"
    SYSTEM = "system"
    VISION = "vision"      # Image/video processing
    CUSTOM = "custom"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "int", "float", "bool", "list", "dict"
    description: str
    required: bool = True
    default: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
            "default": self.default,
        }


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }


@dataclass
class Tool:
    """Definition of a tool that Dheera can use."""
    name: str
    description: str
    function: Callable
    category: ToolCategory = ToolCategory.CUSTOM
    parameters: List[ToolParameter] = field(default_factory=list)
    requires_approval: bool = False
    enabled: bool = True
    is_async: bool = False
    action_id: Optional[int] = None  # Maps to DQN action space
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": [p.to_dict() for p in self.parameters],
            "requires_approval": self.requires_approval,
            "enabled": self.enabled,
            "action_id": self.action_id,
        }
    
    def get_signature(self) -> str:
        """Get a human-readable signature."""
        params = ", ".join([
            f"{p.name}: {p.type}" + ("" if p.required else f" = {p.default}")
            for p in self.parameters
        ])
        return f"{self.name}({params})"
    
    def validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """Validate parameters against schema. Returns error message or None."""
        for param in self.parameters:
            if param.required and param.name not in params:
                return f"Missing required parameter: {param.name}"
        return None


class WebSearchTool:
    """
    Web Search capability using DuckDuckGo.
    Provides search, news, and fact-checking functionality.
    """
    
    def __init__(self, max_results: int = 5, timeout: int = 10, cache_ttl: int = 300):
        self.max_results = max_results
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, dict] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self.search_history: List[dict] = []
        self.total_searches = 0
        self._ddgs = None
    
    def _get_cached(self, query: str) -> Optional[dict]:
        """Get cached result if still valid."""
        key = query.lower().strip()
        if key in self.cache:
            if time.time() - self.cache_timestamps[key] < self.cache_ttl:
                result = self.cache[key].copy()
                result['from_cache'] = True
                return result
            else:
                del self.cache[key]
                del self.cache_timestamps[key]
        return None
    
    def _set_cache(self, query: str, result: dict):
        """Cache a search result."""
        key = query.lower().strip()
        self.cache[key] = result
        self.cache_timestamps[key] = time.time()
    
    def search(self, query: str, max_results: Optional[int] = None, use_cache: bool = True) -> dict:
        """
        Search the web using DuckDuckGo.
        
        Args:
            query: Search query string
            max_results: Override default max results
            use_cache: Whether to use cached results
            
        Returns:
            dict with results and metadata
        """
        # Check cache
        if use_cache:
            cached = self._get_cached(query)
            if cached:
                return cached
        
        results_limit = max_results or self.max_results
        
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    max_results=results_limit,
                    safesearch='moderate'
                ))
            
            formatted = []
            for r in results:
                formatted.append({
                    'title': r.get('title', ''),
                    'url': r.get('href', ''),
                    'snippet': r.get('body', ''),
                })
            
            search_record = {
                'query': query,
                'timestamp': time.time(),
                'result_count': len(formatted),
                'results': formatted,
                'success': True,
                'from_cache': False
            }
            
            self._set_cache(query, search_record)
            self.search_history.append({
                'query': query,
                'timestamp': search_record['timestamp'],
                'result_count': len(formatted)
            })
            self.total_searches += 1
            
            return search_record
            
        except ImportError:
            return {
                'query': query,
                'success': False,
                'error': 'duckduckgo-search not installed. Run: pip install duckduckgo-search',
                'results': []
            }
        except Exception as e:
            return {
                'query': query,
                'timestamp': time.time(),
                'success': False,
                'error': str(e),
                'results': [],
                'from_cache': False
            }
    
    def search_news(self, query: str, max_results: Optional[int] = None) -> dict:
        """Search recent news articles."""
        results_limit = max_results or self.max_results
        
        try:
            from duckduckgo_search import DDGS
            
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
                'timestamp': time.time(),
                'result_count': len(formatted),
                'results': formatted,
                'success': True,
                'type': 'news'
            }
            
        except ImportError:
            return {'success': False, 'error': 'duckduckgo-search not installed', 'results': []}
        except Exception as e:
            return {'success': False, 'error': str(e), 'results': [], 'type': 'news'}
    
    def fact_check(self, claim: str) -> dict:
        """
        Verify a claim by searching multiple sources.
        
        Returns search results with evidence from multiple queries.
        """
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
            'timestamp': time.time(),
            'evidence_count': len(unique_results),
            'evidence': unique_results[:self.max_results],
            'success': True
        }
    
    def get_page_content(self, url: str, max_length: int = 4000) -> dict:
        """Fetch and extract text from a URL."""
        try:
            import httpx
            import re
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, headers=headers, follow_redirects=True)
                response.raise_for_status()
            
            content = response.text
            
            # Remove scripts and styles
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
    
    def format_for_context(self, result: dict) -> str:
        """Format search results as context for SLM."""
        results = result.get('results') or result.get('evidence', [])
        query = result.get('query') or result.get('claim', 'search')
        
        lines = [f"[Web Search: {query}]\n"]
        
        for i, r in enumerate(results[:5], 1):
            lines.append(f"{i}. {r['title']}")
            snippet = r.get('snippet', '')[:200]
            lines.append(f"   {snippet}...")
            lines.append(f"   URL: {r['url']}\n")
        
        return '\n'.join(lines)
    
    def get_stats(self) -> dict:
        """Get search statistics."""
        return {
            'total_searches': self.total_searches,
            'cache_size': len(self.cache),
            'recent_queries': [h['query'] for h in self.search_history[-5:]]
        }


class ToolRegistry:
    """
    Registry for managing Dheera's tools.
    
    Features:
    - Register/unregister tools
    - Execute tools with validation
    - Generate tool descriptions for SLM
    - Track tool usage statistics
    - Web search integration
    - DQN action mapping
    """
    
    # Action IDs mapping to DQN
    ACTION_USE_TOOL = 2
    ACTION_SEARCH_WEB = 3
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.tools: Dict[str, Tool] = {}
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        self.action_mapping: Dict[int, str] = {}  # action_id -> tool_name
        
        # Initialize web search
        web_config = self.config.get('tools', {}).get('web_search', {})
        self.web_search = WebSearchTool(
            max_results=web_config.get('max_results', 5),
            timeout=web_config.get('timeout', 10),
            cache_ttl=web_config.get('cache_ttl', 300)
        )
        
        # Register built-in tools
        self._register_builtins()
    
    def _register_builtins(self):
        """Register built-in utility tools."""
        
        # Calculator tool
        self.register(
            name="calculator",
            description="Evaluate mathematical expressions safely",
            function=self._builtin_calculator,
            category=ToolCategory.UTILITY,
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Mathematical expression to evaluate (e.g., '2 + 2 * 3')"
                )
            ]
        )
        
        # Current time tool
        self.register(
            name="get_time",
            description="Get the current date and time",
            function=self._builtin_get_time,
            category=ToolCategory.UTILITY,
            parameters=[]
        )
        
        # JSON formatter tool
        self.register(
            name="format_json",
            description="Format and validate JSON strings",
            function=self._builtin_format_json,
            category=ToolCategory.UTILITY,
            parameters=[
                ToolParameter(
                    name="json_string",
                    type="string",
                    description="JSON string to format"
                )
            ]
        )
        
        # Web Search tool
        self.register(
            name="web_search",
            description="Search the internet for current information",
            function=self._web_search_wrapper,
            category=ToolCategory.SEARCH,
            action_id=self.ACTION_SEARCH_WEB,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query"
                ),
                ToolParameter(
                    name="max_results",
                    type="int",
                    description="Maximum number of results",
                    required=False,
                    default=5
                )
            ]
        )
        
        # News Search tool
        self.register(
            name="search_news",
            description="Search for recent news articles",
            function=self._news_search_wrapper,
            category=ToolCategory.SEARCH,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="News search query"
                ),
                ToolParameter(
                    name="max_results",
                    type="int",
                    description="Maximum number of results",
                    required=False,
                    default=5
                )
            ]
        )
        
        # Fact Check tool
        self.register(
            name="fact_check",
            description="Verify a claim by searching multiple sources",
            function=self._fact_check_wrapper,
            category=ToolCategory.SEARCH,
            parameters=[
                ToolParameter(
                    name="claim",
                    type="string",
                    description="The claim to verify"
                )
            ]
        )
        
        # Fetch URL content
        self.register(
            name="fetch_url",
            description="Fetch and extract text content from a URL",
            function=self._fetch_url_wrapper,
            category=ToolCategory.SEARCH,
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="URL to fetch"
                ),
                ToolParameter(
                    name="max_length",
                    type="int",
                    description="Maximum content length",
                    required=False,
                    default=4000
                )
            ]
        )
    
    # ==================== Built-in Functions ====================
    
    def _builtin_calculator(self, expression: str) -> float:
        """Safe mathematical expression evaluator."""
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Invalid characters in expression")
        
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return float(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate: {e}")
    
    def _builtin_get_time(self) -> str:
        """Get current time."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _builtin_format_json(self, json_string: str) -> str:
        """Format JSON string."""
        try:
            data = json.loads(json_string)
            return json.dumps(data, indent=2)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    
    # ==================== Web Search Wrappers ====================
    
    def _web_search_wrapper(self, query: str, max_results: int = 5) -> dict:
        """Wrapper for web search."""
        result = self.web_search.search(query, max_results=max_results)
        if result['success']:
            result['formatted_context'] = self.web_search.format_for_context(result)
        return result
    
    def _news_search_wrapper(self, query: str, max_results: int = 5) -> dict:
        """Wrapper for news search."""
        result = self.web_search.search_news(query, max_results=max_results)
        if result['success']:
            result['formatted_context'] = self.web_search.format_for_context(result)
        return result
    
    def _fact_check_wrapper(self, claim: str) -> dict:
        """Wrapper for fact checking."""
        result = self.web_search.fact_check(claim)
        if result['success']:
            result['formatted_context'] = self.web_search.format_for_context(result)
        return result
    
    def _fetch_url_wrapper(self, url: str, max_length: int = 4000) -> dict:
        """Wrapper for URL fetching."""
        return self.web_search.get_page_content(url, max_length=max_length)
    
    # ==================== Registry Methods ====================
    
    def register(
        self,
        name: str,
        description: str,
        function: Callable,
        category: ToolCategory = ToolCategory.CUSTOM,
        parameters: Optional[List[ToolParameter]] = None,
        requires_approval: bool = False,
        is_async: bool = False,
        action_id: Optional[int] = None,
        **metadata
    ) -> Tool:
        """
        Register a new tool.
        
        Args:
            name: Unique tool name
            description: What the tool does
            function: The callable to execute
            category: Tool category
            parameters: Parameter definitions
            requires_approval: Whether human approval is needed
            is_async: Whether the function is async
            action_id: DQN action ID mapping
            **metadata: Additional metadata
            
        Returns:
            The registered Tool
        """
        if parameters is None:
            parameters = self._extract_parameters(function)
        
        tool = Tool(
            name=name,
            description=description,
            function=function,
            category=category,
            parameters=parameters,
            requires_approval=requires_approval,
            is_async=is_async,
            action_id=action_id,
            metadata=metadata,
        )
        
        self.tools[name] = tool
        self.usage_stats[name] = {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "total_time": 0.0,
        }
        
        # Map action ID to tool
        if action_id is not None:
            self.action_mapping[action_id] = name
        
        return tool
    
    def _extract_parameters(self, func: Callable) -> List[ToolParameter]:
        """Extract parameters from function signature."""
        params = []
        sig = inspect.signature(func)
        
        for name, param in sig.parameters.items():
            if name in ('self', 'cls'):
                continue
            
            type_str = "string"
            if param.annotation != inspect.Parameter.empty:
                type_map = {
                    str: "string",
                    int: "int",
                    float: "float",
                    bool: "bool",
                    list: "list",
                    dict: "dict",
                }
                type_str = type_map.get(param.annotation, "string")
            
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default
            
            params.append(ToolParameter(
                name=name,
                type=type_str,
                description=f"Parameter: {name}",
                required=required,
                default=default,
            ))
        
        return params
    
    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self.tools:
            tool = self.tools[name]
            if tool.action_id is not None and tool.action_id in self.action_mapping:
                del self.action_mapping[tool.action_id]
            del self.tools[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def get_by_action_id(self, action_id: int) -> Optional[Tool]:
        """Get a tool by its DQN action ID."""
        tool_name = self.action_mapping.get(action_id)
        if tool_name:
            return self.tools.get(tool_name)
        return None
    
    def execute(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> ToolResult:
        """
        Execute a tool.
        
        Args:
            name: Tool name
            params: Parameters to pass
            timeout: Execution timeout in seconds
            
        Returns:
            ToolResult with output or error
        """
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool not found: {name}"
            )
        
        if not tool.enabled:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool is disabled: {name}"
            )
        
        params = params or {}
        
        # Validate parameters
        validation_error = tool.validate_params(params)
        if validation_error:
            return ToolResult(
                success=False,
                output=None,
                error=validation_error
            )
        
        # Execute
        start_time = time.time()
        try:
            if tool.is_async:
                loop = asyncio.get_event_loop()
                output = loop.run_until_complete(tool.function(**params))
            else:
                output = tool.function(**params)
            
            execution_time = time.time() - start_time
            
            # Update stats
            self.usage_stats[name]["calls"] += 1
            self.usage_stats[name]["successes"] += 1
            self.usage_stats[name]["total_time"] += execution_time
            
            return ToolResult(
                success=True,
                output=output,
                execution_time=execution_time,
                metadata={"tool_name": name}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Update stats
            self.usage_stats[name]["calls"] += 1
            self.usage_stats[name]["failures"] += 1
            self.usage_stats[name]["total_time"] += execution_time
            
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=execution_time,
                metadata={"tool_name": name}
            )
    
    def execute_by_action_id(
        self,
        action_id: int,
        context: Dict[str, Any],
    ) -> ToolResult:
        """
        Execute a tool by its DQN action ID.
        
        Args:
            action_id: DQN action ID
            context: Execution context with user_message, query, etc.
            
        Returns:
            ToolResult
        """
        tool_name = self.action_mapping.get(action_id)
        if not tool_name:
            return ToolResult(
                success=False,
                output=None,
                error=f"No tool mapped to action ID: {action_id}"
            )
        
        # Build params from context
        params = self._extract_params_from_context(tool_name, context)
        return self.execute(tool_name, params)
    
    def _extract_params_from_context(
        self,
        tool_name: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract tool parameters from conversation context."""
        tool = self.get(tool_name)
        if not tool:
            return {}
        
        params = {}
        
        # Handle web search specifically
        if tool_name in ('web_search', 'search_news'):
            query = context.get('query') or context.get('user_message', '')
            # Clean query for search
            params['query'] = self._clean_search_query(query)
            if 'max_results' in context:
                params['max_results'] = context['max_results']
        
        elif tool_name == 'fact_check':
            params['claim'] = context.get('claim') or context.get('user_message', '')
        
        elif tool_name == 'fetch_url':
            params['url'] = context.get('url', '')
        
        else:
            # Generic parameter extraction
            for param in tool.parameters:
                if param.name in context:
                    params[param.name] = context[param.name]
                elif not param.required:
                    params[param.name] = param.default
        
        return params
    
    def _clean_search_query(self, text: str) -> str:
        """Clean and optimize text for web search."""
        stop_words = {
            'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are',
            'the', 'a', 'an', 'can', 'could', 'would', 'should', 'do',
            'does', 'did', 'please', 'tell', 'me', 'about', 'explain',
            'help', 'want', 'need', 'know', 'find', 'search', 'look'
        }
        
        words = text.lower().split()
        key_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        if len(key_words) < 2:
            return text
        
        return ' '.join(key_words[:8])
    
    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        enabled_only: bool = True,
    ) -> List[Tool]:
        """List available tools."""
        tools = list(self.tools.values())
        
        if category:
            tools = [t for t in tools if t.category == category]
        
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        
        return tools
    
    def get_tools_description(self) -> str:
        """Generate tool descriptions for SLM prompt."""
        lines = ["Available tools:"]
        
        for tool in self.list_tools():
            lines.append(f"\n- {tool.name}: {tool.description}")
            lines.append(f"  Usage: {tool.get_signature()}")
            
            if tool.requires_approval:
                lines.append("  ⚠️ Requires human approval")
            
            if tool.action_id is not None:
                lines.append(f"  Action ID: {tool.action_id}")
        
        return "\n".join(lines)
    
    def get_tool_schema(self) -> List[Dict[str, Any]]:
        """Get tool schemas in OpenAI function-calling format."""
        schemas = []
        
        for tool in self.list_tools():
            schema = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            }
            
            for param in tool.parameters:
                schema["parameters"]["properties"][param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.required:
                    schema["parameters"]["required"].append(param.name)
            
            schemas.append(schema)
        
        return schemas
    
    def get_search_tool(self) -> WebSearchTool:
        """Get direct access to the web search tool."""
        return self.web_search
    
    def should_search(self, message: str) -> bool:
        """
        Determine if a message should trigger web search.
        Used by DQN to inform action selection.
        """
        message_lower = message.lower()
        
        search_triggers = [
            'search', 'find', 'look up', 'google', 'what is',
            'who is', 'latest', 'current', 'news', 'today',
            'recent', 'update', 'how to', 'where is', 'when did',
            'price of', 'weather', 'score', 'result', 'happening'
        ]
        
        return any(trigger in message_lower for trigger in search_triggers)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all tools."""
        return {
            "total_tools": len(self.tools),
            "enabled_tools": len([t for t in self.tools.values() if t.enabled]),
            "search_tools": len([t for t in self.tools.values() if t.category == ToolCategory.SEARCH]),
            "web_search_stats": self.web_search.get_stats(),
            "tool_stats": self.usage_stats,
            "action_mapping": self.action_mapping,
        }


# ==================== Test ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Dheera Tool Registry with Web Search")
    print("=" * 60)
    
    # Initialize with config
    config = {
        'tools': {
            'web_search': {
                'max_results': 3,
                'timeout': 10,
                'cache_ttl': 300
            }
        }
    }
    
    registry = ToolRegistry(config)
    
    # Test built-in tools
    print("\n📊 Testing Built-in Tools:")
    print("-" * 40)
    
    print("\n1. Calculator:")
    result = registry.execute("calculator", {"expression": "2 + 2 * 3"})
    print(f"   2 + 2 * 3 = {result.output}")
    
    print("\n2. Get Time:")
    result = registry.execute("get_time", {})
    print(f"   Current time: {result.output}")
    
    print("\n3. Format JSON:")
    result = registry.execute("format_json", {"json_string": '{"name":"Dheera","version":"0.2.0"}'})
    print(f"   Formatted:\n{result.output}")
    
    # Test web search
    print("\n🔍 Testing Web Search Tools:")
    print("-" * 40)
    
    print("\n4. Web Search:")
    result = registry.execute("web_search", {"query": "Python programming"})
    if result.success:
        output = result.output
        print(f"   Found {output.get('result_count', 0)} results")
        for r in output.get('results', [])[:2]:
            print(f"   - {r['title'][:50]}...")
    else:
        print(f"   Error: {result.error}")
    
    print("\n5. News Search:")
    result = registry.execute("search_news", {"query": "artificial intelligence"})
    if result.success:
        output = result.output
        print(f"   Found {output.get('result_count', 0)} news articles")
    else:
        print(f"   Error: {result.error}")
    
    print("\n6. Fact Check:")
    result = registry.execute("fact_check", {"claim": "Python was created by Guido van Rossum"})
    if result.success:
        output = result.output
        print(f"   Found {output.get('evidence_count', 0)} pieces of evidence")
    else:
        print(f"   Error: {result.error}")
    
    # Test action ID execution
    print("\n🎯 Testing DQN Action Mapping:")
    print("-" * 40)
    
    print("\n7. Execute by Action ID (3 = SEARCH_WEB):")
    result = registry.execute_by_action_id(3, {
        'user_message': 'What is the latest version of Ollama?'
    })
    if result.success:
        print(f"   Success! Query used: {result.output.get('query', 'N/A')}")
        print(f"   Results: {result.output.get('result_count', 0)}")
    else:
        print(f"   Error: {result.error}")
    
    # Test search trigger detection
    print("\n8. Search Trigger Detection:")
    test_messages = [
        "What is the latest news about AI?",
        "Hello, how are you?",
        "Search for Python tutorials",
        "Tell me a joke",
        "Who is the president of USA?"
    ]
    for msg in test_messages:
        should_search = registry.should_search(msg)
        icon = "🔍" if should_search else "💬"
        print(f"   {icon} '{msg[:40]}...' -> Search: {should_search}")
    
    # List all tools
    print("\n📋 Registered Tools:")
    print("-" * 40)
    print(registry.get_tools_description())
    
    # Stats
    print("\n📈 Usage Statistics:")
    print("-" * 40)
    stats = registry.get_stats()
    print(f"   Total tools: {stats['total_tools']}")
    print(f"   Search tools: {stats['search_tools']}")
    print(f"   Web searches: {stats['web_search_stats']['total_searches']}")
    print(f"   Action mapping: {stats['action_mapping']}")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
