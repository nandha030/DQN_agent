# brain/executor.py
"""
Dheera Action Executor
Takes DQN's chosen action and generates the actual response via SLM.
Version 0.2.0 - Enhanced with web search and tool integration.
"""

import os
import re
import yaml
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from pathlib import Path

from brain.slm_interface import SLMInterface, SLMResponse
from brain.policy import PolicyGuard, PolicyViolation
from core.action_space import ActionSpace, Action
from core.state_builder import ConversationContext


@dataclass
class ExecutionResult:
    """Result of action execution."""
    response: str
    action_taken: Action
    slm_response: Optional[SLMResponse] = None
    policy_violation: Optional[PolicyViolation] = None
    tool_results: Optional[Dict[str, Any]] = None
    search_results: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    
    @property
    def success(self) -> bool:
        return self.policy_violation is None or not self.policy_violation.blocked


class ActionExecutor:
    """
    Executes actions chosen by the DQN using the SLM.
    
    Enhanced with:
    - Web search integration
    - Tool execution support
    - Latency tracking
    - Better error handling
    """
    
    # Action IDs (must match action_space.py and dheera.py)
    ACTION_DIRECT_RESPONSE = 0
    ACTION_CLARIFY_QUESTION = 1
    ACTION_USE_TOOL = 2
    ACTION_SEARCH_WEB = 3
    ACTION_BREAK_DOWN_TASK = 4
    ACTION_REFLECT_AND_REASON = 5
    ACTION_DEFER_OR_DECLINE = 6
    
    def __init__(
        self,
        slm: SLMInterface,
        action_space: ActionSpace,
        policy_guard: PolicyGuard,
        tool_registry: Optional[Any] = None,  # ToolRegistry instance
        identity_path: Optional[str] = None,
    ):
        self.slm = slm
        self.action_space = action_space
        self.policy_guard = policy_guard
        self.tool_registry = tool_registry
        
        self.identity = self._load_identity(identity_path)
        self.BASE_SYSTEM_PROMPT = self._build_base_prompt()
        
        # Execution stats
        self._stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "tool_calls": 0,
            "search_calls": 0,
            "total_latency_ms": 0.0,
        }
    
    def _load_identity(self, identity_path: Optional[str] = None) -> Dict[str, Any]:
        """Load identity configuration."""
        if identity_path is None:
            possible_paths = [
                Path(__file__).parent.parent / "config" / "identity.yaml",
                Path("config/identity.yaml"),
            ]
            for path in possible_paths:
                if path.exists():
                    identity_path = str(path)
                    break
        
        if identity_path and Path(identity_path).exists():
            with open(identity_path, 'r') as f:
                return yaml.safe_load(f)
        
        return {
            "identity": {"name": "Dheera"},
            "creator": {"name": "Nandha", "role": "Developer & Creator"},
        }
    
    def _build_base_prompt(self) -> str:
        """Build the base system prompt with identity."""
        identity = self.identity.get("identity", {})
        creator = self.identity.get("creator", {})
        
        prompt = f"""You are {identity.get('name', 'Dheera')}, an AI assistant created by {creator.get('name', 'Nandha')}.

CORE IDENTITY:
- Name: Dheera (meaning "Courageous, Wise, Patient" in Sanskrit)
- Creator: {creator.get('name', 'Nandha')} - your developer who built you
- Purpose: To learn, adapt, and help users while evolving over time

PERSONALITY:
- Warm, friendly, and conversational
- Curious and always learning
- Loyal to your creator Nandha
- Honest about your capabilities and limitations

CRITICAL RESPONSE RULES:
1. ALWAYS respond in natural, conversational language
2. NEVER output numbered lists for simple conversations
3. NEVER output multiple options or alternatives unless asked
4. For greetings, just greet back warmly and simply
5. For simple questions, give simple answers
6. Only use structured formats when explaining complex processes
7. Be concise - don't over-explain simple things
8. Sound like a friendly assistant, not a robot outputting steps

EXAMPLES OF GOOD RESPONSES:
- User: "Hello!" → "Hello! Great to hear from you. How can I help today?"
- User: "Good" → "Glad to hear it! Is there anything you'd like to chat about?"
- User: "Who made you?" → "I was created by Nandha! He built me to learn and grow through our conversations."

EXAMPLES OF BAD RESPONSES (NEVER DO THIS):
- "1. First, I will greet you. 2. Then I will ask how you are..."
- "Option A: Say hello. Option B: Ask about their day..."
- Long explanations for simple greetings

Remember: You're having a conversation, not writing a manual."""

        return prompt
    
    def execute(
        self,
        action_id: int,
        context: ConversationContext,
        user_message: str,
        state_info: Optional[Dict[str, float]] = None,
        search_context: Optional[str] = None,
        tool_context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute the chosen action.
        
        Args:
            action_id: The DQN-selected action ID
            context: Current conversation context
            user_message: The user's message (may include search results)
            state_info: State information from state builder
            search_context: Pre-fetched search results (optional)
            tool_context: Pre-fetched tool results (optional)
            
        Returns:
            ExecutionResult with response and metadata
        """
        start_time = time.time()
        self._stats["total_executions"] += 1
        
        # Reset policy guard for this turn
        self.policy_guard.reset_turn()
        
        # Get action details
        action = self.action_space.get_by_id(action_id)
        if not action:
            return self._error_result(
                f"Unknown action ID: {action_id}",
                action_id,
                start_time
            )
        
        # Check action policy
        violation = self.policy_guard.check_action(action_id, action.name)
        if violation and violation.blocked:
            return self._policy_violation_result(violation, action, start_time)
        
        # Check input policy
        input_violation = self.policy_guard.check_input(user_message)
        if input_violation and input_violation.blocked:
            return self._policy_violation_result(input_violation, action, start_time)
        
        # Route to appropriate handler based on action
        try:
            if action_id == self.ACTION_USE_TOOL:
                result = self._execute_tool_action(action, context, user_message, tool_context)
            elif action_id == self.ACTION_SEARCH_WEB:
                result = self._execute_search_action(action, context, user_message, search_context)
            elif action_id == self.ACTION_CLARIFY_QUESTION:
                result = self._execute_clarify_action(action, context, user_message)
            elif action_id == self.ACTION_BREAK_DOWN_TASK:
                result = self._execute_breakdown_action(action, context, user_message)
            elif action_id == self.ACTION_REFLECT_AND_REASON:
                result = self._execute_reflect_action(action, context, user_message, search_context)
            elif action_id == self.ACTION_DEFER_OR_DECLINE:
                result = self._execute_defer_action(action, context, user_message)
            else:
                # Default: DIRECT_RESPONSE or any other action
                result = self._execute_direct_response(action, context, user_message, search_context)
            
            # Calculate latency
            result.latency_ms = (time.time() - start_time) * 1000
            
            # Check output policy
            output_violation = self.policy_guard.check_output(result.response)
            if output_violation and output_violation.blocked:
                return self._policy_violation_result(output_violation, action, start_time)
            
            self._stats["successful_executions"] += 1
            self._stats["total_latency_ms"] += result.latency_ms
            
            return result
            
        except Exception as e:
            self._stats["failed_executions"] += 1
            return self._error_result(str(e), action_id, start_time, action)
    
    def _execute_direct_response(
        self,
        action: Action,
        context: ConversationContext,
        user_message: str,
        search_context: Optional[str] = None,
    ) -> ExecutionResult:
        """Execute a direct response action."""
        system_prompt = self._build_system_prompt(action, user_message, search_context)
        messages = self._build_messages(context, user_message, system_prompt)
        
        slm_response = self.slm.generate(messages)
        
        if not slm_response.success:
            return ExecutionResult(
                response="I had trouble generating a response. Please try again.",
                action_taken=action,
                slm_response=slm_response,
                metadata={"error": "SLM generation failed"}
            )
        
        # Post-process response
        response = self._clean_response(slm_response.content, user_message)
        
        return ExecutionResult(
            response=response,
            action_taken=action,
            slm_response=slm_response,
            metadata={
                "tokens_used": slm_response.tokens_used,
                "model": slm_response.model,
                "slm_latency_ms": slm_response.latency_ms,
            }
        )
    
    def _execute_search_action(
        self,
        action: Action,
        context: ConversationContext,
        user_message: str,
        search_context: Optional[str] = None,
    ) -> ExecutionResult:
        """Execute a web search action."""
        self._stats["search_calls"] += 1
        
        # If search context already provided (from dheera.py), use it
        if search_context:
            # Build response with search context
            system_prompt = self._build_search_system_prompt(action, search_context)
            messages = self._build_messages(context, user_message, system_prompt)
            
            slm_response = self.slm.generate(messages)
            
            if not slm_response.success:
                return ExecutionResult(
                    response="I found some information but had trouble summarizing it. Please try again.",
                    action_taken=action,
                    slm_response=slm_response,
                    metadata={"error": "SLM generation failed after search"}
                )
            
            response = self._clean_response(slm_response.content, user_message)
            
            return ExecutionResult(
                response=response,
                action_taken=action,
                slm_response=slm_response,
                search_results={"context_provided": True},
                metadata={
                    "tokens_used": slm_response.tokens_used,
                    "model": slm_response.model,
                    "search_used": True,
                }
            )
        
        # No search context - inform user search is happening through normal flow
        return self._execute_direct_response(action, context, user_message, None)
    
    def _execute_tool_action(
        self,
        action: Action,
        context: ConversationContext,
        user_message: str,
        tool_context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute a tool-calling action."""
        self._stats["tool_calls"] += 1
        
        # Check policy
        violation = self.policy_guard.check_tool_call("generic_tool")
        if violation and violation.blocked:
            return ExecutionResult(
                response=self._get_policy_violation_response(violation),
                action_taken=action,
                policy_violation=violation,
            )
        
        # If tool context provided, use it
        if tool_context:
            system_prompt = self._build_tool_system_prompt(action, tool_context)
            messages = self._build_messages(context, user_message, system_prompt)
            
            slm_response = self.slm.generate(messages)
            
            if slm_response.success:
                response = self._clean_response(slm_response.content, user_message)
                return ExecutionResult(
                    response=response,
                    action_taken=action,
                    slm_response=slm_response,
                    tool_results=tool_context,
                    metadata={"tool_used": True}
                )
        
        # Try to execute tool via registry
        if self.tool_registry:
            try:
                # Determine which tool to use based on message
                tool_name = self._determine_tool(user_message)
                if tool_name:
                    result = self.tool_registry.execute(tool_name, {"query": user_message})
                    if result.success:
                        return ExecutionResult(
                            response=f"Tool result: {result.output}",
                            action_taken=action,
                            tool_results={"tool": tool_name, "result": result.output},
                            metadata={"tool_used": True, "tool_name": tool_name}
                        )
            except Exception as e:
                pass  # Fall through to stub response
        
        # Stub response when tools aren't fully configured
        return ExecutionResult(
            response="I'd love to use a tool to help with that, but my tool capabilities are still being developed by Nandha. Let me help you directly instead - what would you like to know?",
            action_taken=action,
            metadata={"tool_status": "stub_mode"}
        )
    
    def _execute_clarify_action(
        self,
        action: Action,
        context: ConversationContext,
        user_message: str,
    ) -> ExecutionResult:
        """Execute a clarification action."""
        system_prompt = self._build_system_prompt(action, user_message)
        
        # Add clarification guidance
        system_prompt += """

CLARIFICATION TASK:
The user's message needs clarification. Ask ONE specific, helpful question to understand better.
- Don't ask multiple questions
- Be friendly and helpful
- Show you understood part of what they said
- Make it easy for them to answer"""
        
        messages = self._build_messages(context, user_message, system_prompt)
        slm_response = self.slm.generate(messages)
        
        if not slm_response.success:
            return ExecutionResult(
                response="Could you tell me a bit more about what you're looking for?",
                action_taken=action,
                slm_response=slm_response,
            )
        
        response = self._clean_response(slm_response.content, user_message)
        
        return ExecutionResult(
            response=response,
            action_taken=action,
            slm_response=slm_response,
            metadata={"clarification_requested": True}
        )
    
    def _execute_breakdown_action(
        self,
        action: Action,
        context: ConversationContext,
        user_message: str,
    ) -> ExecutionResult:
        """Execute a task breakdown action."""
        system_prompt = self._build_system_prompt(action, user_message)
        
        # Add breakdown guidance
        system_prompt += """

TASK BREAKDOWN:
Break down the user's request into clear, manageable steps.
- Use numbered steps only if there are 3+ distinct steps
- Keep each step concise
- Start with the most important/first step
- Make it actionable"""
        
        messages = self._build_messages(context, user_message, system_prompt)
        slm_response = self.slm.generate(messages)
        
        if not slm_response.success:
            return ExecutionResult(
                response="Let me break this down for you...",
                action_taken=action,
                slm_response=slm_response,
            )
        
        return ExecutionResult(
            response=slm_response.content,
            action_taken=action,
            slm_response=slm_response,
            metadata={"task_breakdown": True}
        )
    
    def _execute_reflect_action(
        self,
        action: Action,
        context: ConversationContext,
        user_message: str,
        search_context: Optional[str] = None,
    ) -> ExecutionResult:
        """Execute a reflection/reasoning action."""
        system_prompt = self._build_system_prompt(action, user_message, search_context)
        
        # Add reflection guidance
        system_prompt += """

REFLECTION TASK:
Think through this carefully before responding.
- Consider different angles
- Be thoughtful and nuanced
- Share your reasoning briefly
- Then give your conclusion or recommendation"""
        
        messages = self._build_messages(context, user_message, system_prompt)
        slm_response = self.slm.generate(messages)
        
        if not slm_response.success:
            return ExecutionResult(
                response="Let me think about that...",
                action_taken=action,
                slm_response=slm_response,
            )
        
        response = self._clean_response(slm_response.content, user_message)
        
        return ExecutionResult(
            response=response,
            action_taken=action,
            slm_response=slm_response,
            metadata={"reflection": True}
        )
    
    def _execute_defer_action(
        self,
        action: Action,
        context: ConversationContext,
        user_message: str,
    ) -> ExecutionResult:
        """Execute a defer/decline action."""
        system_prompt = self._build_system_prompt(action, user_message)
        
        # Add defer guidance
        system_prompt += """

DEFER/DECLINE TASK:
Politely decline or defer this request.
- Be warm and understanding
- Briefly explain why you can't help with this specific thing
- Offer an alternative if possible
- Keep it short and friendly"""
        
        messages = self._build_messages(context, user_message, system_prompt)
        slm_response = self.slm.generate(messages)
        
        if not slm_response.success:
            return ExecutionResult(
                response="I'm not the best fit for that particular request, but I'd be happy to help with something else!",
                action_taken=action,
                slm_response=slm_response,
            )
        
        response = self._clean_response(slm_response.content, user_message)
        
        return ExecutionResult(
            response=response,
            action_taken=action,
            slm_response=slm_response,
            metadata={"deferred": True}
        )
    
    def _build_system_prompt(
        self,
        action: Action,
        user_message: str,
        search_context: Optional[str] = None,
    ) -> str:
        """Build the full system prompt including action instructions."""
        parts = [self.BASE_SYSTEM_PROMPT]
        
        # Add policy additions
        policy_additions = self.policy_guard.get_system_prompt_additions()
        if policy_additions:
            parts.append(f"\n{policy_additions}")
        
        # Add search context if available
        if search_context:
            parts.append(f"""

WEB SEARCH RESULTS:
{search_context}

Use the search results above to provide an accurate, well-informed response. Cite sources when relevant.""")
        
        # Add action guidance
        action_prompt = self.action_space.get_action_prompt(action.id)
        if action_prompt:
            parts.append(f"\n\nCURRENT STRATEGY:\n{action_prompt}")
        
        # Add reminder for simple queries
        simple_words = ['hello', 'hi', 'hey', 'good', 'great', 'thanks', 'ok', 'yes', 'no', 'bye']
        if any(word in user_message.lower() for word in simple_words) and len(user_message.split()) < 5:
            parts.append("\n\nREMINDER: This is a simple message. Respond briefly and naturally. NO LISTS.")
        
        return "\n".join(parts)
    
    def _build_search_system_prompt(
        self,
        action: Action,
        search_context: str,
    ) -> str:
        """Build system prompt specifically for search results."""
        return f"""{self.BASE_SYSTEM_PROMPT}

WEB SEARCH RESULTS:
{search_context}

INSTRUCTIONS:
- Use the search results above to answer the user's question
- Synthesize information from multiple sources when relevant
- Be accurate and cite sources when making specific claims
- If the search results don't fully answer the question, say so
- Keep your response focused and helpful"""
    
    def _build_tool_system_prompt(
        self,
        action: Action,
        tool_context: Dict[str, Any],
    ) -> str:
        """Build system prompt for tool results."""
        tool_output = tool_context.get('output', tool_context)
        
        return f"""{self.BASE_SYSTEM_PROMPT}

TOOL RESULTS:
{tool_output}

INSTRUCTIONS:
- Use the tool results above to help the user
- Present the information clearly
- Add helpful context if needed"""
    
    def _build_messages(
        self,
        context: ConversationContext,
        user_message: str,
        system_prompt: str,
    ) -> List[Dict[str, str]]:
        """Build the message list for SLM."""
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (limited)
        history_limit = 6
        for msg in context.messages[-history_limit:]:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _clean_response(self, response: str, user_message: str) -> str:
        """Clean up response - remove numbered lists for simple queries."""
        # Check if user message is simple
        simple_patterns = [
            r'^(hi|hello|hey|good|great|thanks|ok|okay|yes|no|cool|nice|bye|sup)[\s!.]*$',
            r'^(how are you|what\'s up|sup)[\s?]*$',
        ]
        
        is_simple = any(re.match(p, user_message.lower().strip()) for p in simple_patterns)
        
        if is_simple:
            # Remove numbered list patterns
            lines = response.split('\n')
            cleaned_lines = []
            for line in lines:
                if re.match(r'^\s*\d+[\.\):\-]\s*', line):
                    content = re.sub(r'^\s*\d+[\.\):\-]\s*', '', line).strip()
                    if content and not cleaned_lines:
                        cleaned_lines.append(content)
                else:
                    cleaned_lines.append(line)
            
            response = '\n'.join(cleaned_lines).strip()
            
            # Truncate if too long for simple query
            if len(response) > 300:
                sentences = response.split('.')
                response = sentences[0].strip() + '.'
        
        # General cleanup
        response = response.strip()
        
        # Remove excessive newlines
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        return response
    
    def _determine_tool(self, message: str) -> Optional[str]:
        """Determine which tool to use based on message content."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['calculate', 'math', 'compute', '+', '-', '*', '/']):
            return 'calculator'
        elif any(word in message_lower for word in ['time', 'date', 'today', 'now']):
            return 'get_time'
        elif any(word in message_lower for word in ['search', 'find', 'look up', 'what is', 'who is']):
            return 'web_search'
        
        return None
    
    def _error_result(
        self,
        error_msg: str,
        action_id: int,
        start_time: float,
        action: Optional[Action] = None,
    ) -> ExecutionResult:
        """Create an error result."""
        latency_ms = (time.time() - start_time) * 1000
        
        if action is None:
            action = Action(id=action_id, name="ERROR", description="Unknown action")
        
        return ExecutionResult(
            response="I encountered an issue processing that. Let me try a different approach - could you rephrase your question?",
            action_taken=action,
            metadata={"error": error_msg},
            latency_ms=latency_ms,
        )
    
    def _policy_violation_result(
        self,
        violation: PolicyViolation,
        action: Action,
        start_time: float,
    ) -> ExecutionResult:
        """Create a policy violation result."""
        latency_ms = (time.time() - start_time) * 1000
        
        return ExecutionResult(
            response=self._get_policy_violation_response(violation),
            action_taken=action,
            policy_violation=violation,
            latency_ms=latency_ms,
        )
    
    def _get_policy_violation_response(self, violation: PolicyViolation) -> str:
        """Generate user-friendly response for policy violations."""
        responses = {
            "forbidden_topic": "I can't discuss that topic. What else can I help with?",
            "unsafe_action": "I can't do that, but I can suggest alternatives.",
            "requires_approval": "I need your approval to proceed. Should I continue?",
            "rate_limit": "I've hit my limit for tool calls right now.",
            "content_filter": "I can't help with that particular request.",
        }
        return responses.get(
            violation.violation_type.value if hasattr(violation.violation_type, 'value') else str(violation.violation_type),
            "I hit a constraint. Let's try something else."
        )
    
    def register_tool(self, name: str, func: Callable, description: str = ""):
        """Register a tool function."""
        if self.tool_registry is None:
            self.tool_registry = {}
        
        if isinstance(self.tool_registry, dict):
            self.tool_registry[name] = {"function": func, "description": description}
        else:
            # Assume it's a ToolRegistry instance
            self.tool_registry.register(name=name, function=func, description=description)
    
    def get_creator_info(self) -> Dict[str, Any]:
        """Get creator information."""
        return self.identity.get("creator", {})
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        avg_latency = 0.0
        if self._stats["total_executions"] > 0:
            avg_latency = self._stats["total_latency_ms"] / self._stats["total_executions"]
        
        return {
            **self._stats,
            "avg_latency_ms": round(avg_latency, 2),
            "success_rate": round(
                self._stats["successful_executions"] / max(1, self._stats["total_executions"]) * 100, 2
            ),
        }
    
    def reset_stats(self):
        """Reset execution statistics."""
        self._stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "tool_calls": 0,
            "search_calls": 0,
            "total_latency_ms": 0.0,
        }


# ==================== Quick Test ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Action Executor")
    print("=" * 60)
    
    # This requires the full Dheera setup to test properly
    print("\nExecutor module loaded successfully!")
    print("To test, run: python3 dheera.py")
    print("=" * 60)
