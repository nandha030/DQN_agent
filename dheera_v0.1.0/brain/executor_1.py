"""
Dheera Action Executor
Takes DQN's chosen action and generates the actual response via SLM.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Callable
import json

from .slm_interface import SLMInterface, SLMResponse
from .policy import PolicyGuard, PolicyViolation
from ..core.action_space import ActionSpace, Action
from ..core.state_builder import ConversationContext


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
    """
    Executes actions chosen by the DQN using the SLM.
    
    Flow:
    1. DQN selects action (e.g., ASK_CLARIFYING_QUESTION)
    2. Executor builds appropriate prompt for that action
    3. SLM generates the response
    4. Policy guard checks the output
    5. Returns final response to user
    """
    
    # Dheera's core personality and system prompt
    BASE_SYSTEM_PROMPT = """You are Dheera (धीर), an adaptive AI agent designed to learn and evolve.

Your name means "courageous", "wise", and "patient" in Sanskrit.

Core traits:
- You are helpful, curious, and always learning
- You adapt your communication style based on context
- You think before acting and explain your reasoning when helpful
- You are honest about your limitations and uncertainties

You have two "brains":
- A fast brain (DQN) that decides WHAT strategy to use
- A slow brain (SLM - you) that decides HOW to execute that strategy

The fast brain has chosen a strategy for this response. Follow it while maintaining your helpful nature."""

    def __init__(
        self,
        slm: SLMInterface,
        action_space: ActionSpace,
        policy_guard: PolicyGuard,
        tool_registry: Optional[Dict[str, Callable]] = None,
    ):
        self.slm = slm
        self.action_space = action_space
        self.policy_guard = policy_guard
        self.tool_registry = tool_registry or {}
        
    def execute(
        self,
        action_id: int,
        context: ConversationContext,
        user_message: str,
        state_info: Optional[Dict[str, float]] = None,
    ) -> ExecutionResult:
        """
        Execute the chosen action.
        
        Args:
            action_id: Action ID from DQN
            context: Current conversation context
            user_message: The user's latest message
            state_info: Optional state vector breakdown for debugging
            
        Returns:
            ExecutionResult with response and metadata
        """
        # Reset policy turn counter
        self.policy_guard.reset_turn()
        
        # Get action details
        action = self.action_space.get_by_id(action_id)
        if not action:
            return ExecutionResult(
                response="I encountered an internal error. Please try again.",
                action_taken=Action(id=-1, name="ERROR", description="Unknown action"),
                metadata={"error": f"Unknown action ID: {action_id}"}
            )
        
        # Check policy for action
        violation = self.policy_guard.check_action(action_id, action.name)
        if violation and violation.blocked:
            return ExecutionResult(
                response=self._get_policy_violation_response(violation),
                action_taken=action,
                policy_violation=violation,
            )
        
        # Check input policy
        input_violation = self.policy_guard.check_input(user_message)
        if input_violation and input_violation.blocked:
            return ExecutionResult(
                response=self._get_policy_violation_response(input_violation),
                action_taken=action,
                policy_violation=input_violation,
            )
        
        # Handle tool calls specially
        if action.name == "CALL_TOOL" and action.requires_tool:
            return self._execute_tool_action(action, context, user_message)
        
        # Build prompt and generate response
        system_prompt = self._build_system_prompt(action)
        messages = self._build_messages(context, user_message, system_prompt, state_info)
        
        # Generate via SLM
        slm_response = self.slm.generate(messages)
        
        if not slm_response.success:
            return ExecutionResult(
                response="I had trouble generating a response. Please try again.",
                action_taken=action,
                slm_response=slm_response,
                metadata={"error": "SLM generation failed"}
            )
        
        # Check output policy
        output_violation = self.policy_guard.check_output(slm_response.content)
        if output_violation and output_violation.blocked:
            return ExecutionResult(
                response=self._get_policy_violation_response(output_violation),
                action_taken=action,
                policy_violation=output_violation,
                slm_response=slm_response,
            )
        
        return ExecutionResult(
            response=slm_response.content,
            action_taken=action,
            slm_response=slm_response,
            metadata={
                "tokens_used": slm_response.tokens_used,
                "model": slm_response.model,
            }
        )
    
    def _build_system_prompt(self, action: Action) -> str:
        """Build the full system prompt including action instructions."""
        parts = [self.BASE_SYSTEM_PROMPT]
        
        # Add policy constraints
        policy_additions = self.policy_guard.get_system_prompt_additions()
        if policy_additions:
            parts.append(f"\n{policy_additions}")
        
        # Add action-specific instructions
        action_prompt = self.action_space.get_action_prompt(action.id)
        if action_prompt:
            parts.append(f"\n\nCURRENT STRATEGY (chosen by fast brain):\n{action_prompt}")
        
        return "\n".join(parts)
    
    def _build_messages(
        self,
        context: ConversationContext,
        user_message: str,
        system_prompt: str,
        state_info: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, str]]:
        """Build the message list for SLM."""
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last N turns)
        history_limit = 10
        for msg in context.messages[-history_limit:]:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        # Add current message with optional state context
        if state_info:
            # Add state info as context (for debugging/transparency)
            context_note = self._format_state_context(state_info)
            enhanced_message = f"{user_message}\n\n[Internal context: {context_note}]"
        else:
            enhanced_message = user_message
        
        messages.append({"role": "user", "content": enhanced_message})
        
        return messages
    
    def _format_state_context(self, state_info: Dict[str, float]) -> str:
        """Format state info for internal context."""
        relevant = []
        
        if state_info.get("complexity", 0) > 0.5:
            relevant.append("complex query")
        if state_info.get("uncertainty", 0) > 0.3:
            relevant.append("user seems uncertain")
        if state_info.get("frustration", 0) > 0.3:
            relevant.append("user may be frustrated")
        if state_info.get("requires_code", 0) > 0.5:
            relevant.append("likely needs code")
        if state_info.get("requires_math", 0) > 0.5:
            relevant.append("involves math")
        
        return ", ".join(relevant) if relevant else "standard query"
    
    def _execute_tool_action(
        self,
        action: Action,
        context: ConversationContext,
        user_message: str,
    ) -> ExecutionResult:
        """Execute a tool-calling action."""
        # Check rate limit
        violation = self.policy_guard.check_tool_call("generic_tool")
        if violation and violation.blocked:
            return ExecutionResult(
                response=self._get_policy_violation_response(violation),
                action_taken=action,
                policy_violation=violation,
            )
        
        # For now, return a stub response
        # In Phase 1+, this would parse the user message, identify the tool, and call it
        return ExecutionResult(
            response=(
                "I've identified that a tool might help here, but tool execution "
                "is not yet enabled in this version. Let me help you directly instead.\n\n"
                "What specific task would you like me to help with?"
            ),
            action_taken=action,
            metadata={"tool_status": "stub_mode"}
        )
    
    def _get_policy_violation_response(self, violation: PolicyViolation) -> str:
        """Generate user-friendly response for policy violations."""
        responses = {
            "forbidden_topic": (
                "I'm not able to discuss that topic. "
                "Is there something else I can help you with?"
            ),
            "unsafe_action": (
                "I can't perform that action as it's outside my allowed capabilities. "
                "Let me suggest an alternative approach."
            ),
            "requires_approval": (
                "This action requires your explicit approval before I proceed. "
                "Would you like me to continue? (yes/no)"
            ),
            "rate_limit": (
                "I've reached my limit for tool calls this turn. "
                "Let me summarize what we've found so far."
            ),
            "content_filter": (
                "I'm not able to help with that request. "
                "Let's focus on something else I can assist with."
            ),
        }
        
        return responses.get(
            violation.violation_type.value,
            "I encountered a policy constraint. Please try a different request."
        )
    
    def register_tool(self, name: str, func: Callable, description: str = ""):
        """Register a tool for future use."""
        self.tool_registry[name] = {
            "function": func,
            "description": description,
        }


# Quick test
if __name__ == "__main__":
    from ..core.action_space import ActionSpace
    
    # Create components
    slm = SLMInterface({"provider": "ollama", "model": "phi3:mini"})
    action_space = ActionSpace()
    policy = PolicyGuard()
    
    executor = ActionExecutor(slm, action_space, policy)
    
    # Create test context
    context = ConversationContext(
        messages=[],
        turn_count=1,
    )
    
    # Test execution
    result = executor.execute(
        action_id=2,  # GIVE_CONCISE_SUMMARY
        context=context,
        user_message="What is machine learning?",
    )
    
    print(f"Action: {result.action_taken.name}")
    print(f"Response: {result.response[:200]}...")
    print(f"Success: {result.success}")
