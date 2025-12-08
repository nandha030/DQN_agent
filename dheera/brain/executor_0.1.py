"""
Dheera Action Executor
Takes DQN's chosen action and generates the actual response via SLM.
"""

import os
import yaml
from dataclasses import dataclass
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
    metadata: Dict[str, Any] = None
    
    @property
    def success(self) -> bool:
        return self.policy_violation is None or not self.policy_violation.blocked


class ActionExecutor:
    """Executes actions chosen by the DQN using the SLM."""
    
    def __init__(
        self,
        slm: SLMInterface,
        action_space: ActionSpace,
        policy_guard: PolicyGuard,
        tool_registry: Optional[Dict[str, Callable]] = None,
        identity_path: Optional[str] = None,
    ):
        self.slm = slm
        self.action_space = action_space
        self.policy_guard = policy_guard
        self.tool_registry = tool_registry or {}
        
        self.identity = self._load_identity(identity_path)
        self.BASE_SYSTEM_PROMPT = self._build_base_prompt()
    
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
    ) -> ExecutionResult:
        """Execute the chosen action."""
        self.policy_guard.reset_turn()
        
        action = self.action_space.get_by_id(action_id)
        if not action:
            return ExecutionResult(
                response="I encountered an internal error. Please try again.",
                action_taken=Action(id=-1, name="ERROR", description="Unknown action"),
                metadata={"error": f"Unknown action ID: {action_id}"}
            )
        
        violation = self.policy_guard.check_action(action_id, action.name)
        if violation and violation.blocked:
            return ExecutionResult(
                response=self._get_policy_violation_response(violation),
                action_taken=action,
                policy_violation=violation,
            )
        
        input_violation = self.policy_guard.check_input(user_message)
        if input_violation and input_violation.blocked:
            return ExecutionResult(
                response=self._get_policy_violation_response(input_violation),
                action_taken=action,
                policy_violation=input_violation,
            )
        
        if action.name == "CALL_TOOL" and action.requires_tool:
            return self._execute_tool_action(action, context, user_message)
        
        system_prompt = self._build_system_prompt(action, user_message)
        messages = self._build_messages(context, user_message, system_prompt)
        
        slm_response = self.slm.generate(messages)
        
        if not slm_response.success:
            return ExecutionResult(
                response="I had trouble generating a response. Please try again.",
                action_taken=action,
                slm_response=slm_response,
                metadata={"error": "SLM generation failed"}
            )
        
        # Post-process to remove any accidental numbered lists for simple queries
        response = self._clean_response(slm_response.content, user_message)
        
        output_violation = self.policy_guard.check_output(response)
        if output_violation and output_violation.blocked:
            return ExecutionResult(
                response=self._get_policy_violation_response(output_violation),
                action_taken=action,
                policy_violation=output_violation,
                slm_response=slm_response,
            )
        
        return ExecutionResult(
            response=response,
            action_taken=action,
            slm_response=slm_response,
            metadata={
                "tokens_used": slm_response.tokens_used,
                "model": slm_response.model,
            }
        )
    
    def _clean_response(self, response: str, user_message: str) -> str:
        """Clean up response - remove numbered lists for simple queries."""
        import re
        
        # Check if user message is simple (greeting, short feedback, etc.)
        simple_patterns = [
            r'^(hi|hello|hey|good|great|thanks|ok|okay|yes|no|cool|nice)[\s!.]*$',
            r'^(how are you|what\'s up|sup)[\s?]*$',
        ]
        
        is_simple = any(re.match(p, user_message.lower().strip()) for p in simple_patterns)
        
        if is_simple:
            # Remove numbered list patterns from response
            lines = response.split('\n')
            cleaned_lines = []
            for line in lines:
                # Skip lines that start with numbers like "1." or "1:"
                if re.match(r'^\s*\d+[\.\):\-]\s*', line):
                    # Extract the content after the number
                    content = re.sub(r'^\s*\d+[\.\):\-]\s*', '', line).strip()
                    if content and not cleaned_lines:
                        cleaned_lines.append(content)
                else:
                    cleaned_lines.append(line)
            
            response = '\n'.join(cleaned_lines).strip()
            
            # If still too long for a simple query, truncate
            if len(response) > 300:
                sentences = response.split('.')
                response = sentences[0].strip() + '.'
        
        return response
    
    def _build_system_prompt(self, action: Action, user_message: str) -> str:
        """Build the full system prompt including action instructions."""
        parts = [self.BASE_SYSTEM_PROMPT]
        
        policy_additions = self.policy_guard.get_system_prompt_additions()
        if policy_additions:
            parts.append(f"\n{policy_additions}")
        
        # Add action guidance
        action_prompt = self.action_space.get_action_prompt(action.id)
        if action_prompt:
            parts.append(f"\n\nCURRENT STRATEGY:\n{action_prompt}")
        
        # Add reminder for simple queries
        simple_words = ['hello', 'hi', 'hey', 'good', 'great', 'thanks', 'ok', 'yes', 'no']
        if any(word in user_message.lower() for word in simple_words) and len(user_message.split()) < 5:
            parts.append("\n\nREMINDER: This is a simple message. Respond briefly and naturally. NO LISTS.")
        
        return "\n".join(parts)
    
    def _build_messages(
        self,
        context: ConversationContext,
        user_message: str,
        system_prompt: str,
    ) -> List[Dict[str, str]]:
        """Build the message list for SLM."""
        messages = [{"role": "system", "content": system_prompt}]
        
        history_limit = 6
        for msg in context.messages[-history_limit:]:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _execute_tool_action(
        self,
        action: Action,
        context: ConversationContext,
        user_message: str,
    ) -> ExecutionResult:
        """Execute a tool-calling action."""
        violation = self.policy_guard.check_tool_call("generic_tool")
        if violation and violation.blocked:
            return ExecutionResult(
                response=self._get_policy_violation_response(violation),
                action_taken=action,
                policy_violation=violation,
            )
        
        return ExecutionResult(
            response="I'd love to use a tool to help with that, but my tool capabilities are still being developed by Nandha. Let me help you directly instead - what would you like to know?",
            action_taken=action,
            metadata={"tool_status": "stub_mode"}
        )
    
    def _get_policy_violation_response(self, violation: PolicyViolation) -> str:
        """Generate user-friendly response for policy violations."""
        responses = {
            "forbidden_topic": "I can't discuss that topic. What else can I help with?",
            "unsafe_action": "I can't do that, but I can suggest alternatives.",
            "requires_approval": "I need your approval to proceed. Should I continue?",
            "rate_limit": "I've hit my limit for tool calls right now.",
            "content_filter": "I can't help with that request.",
        }
        return responses.get(violation.violation_type.value, "I hit a constraint. Let's try something else.")
    
    def register_tool(self, name: str, func: Callable, description: str = ""):
        self.tool_registry[name] = {"function": func, "description": description}
    
    def get_creator_info(self) -> Dict[str, Any]:
        return self.identity.get("creator", {})
