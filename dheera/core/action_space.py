# core/action_space.py
"""
Dheera Action Space
Defines discrete actions the DQN can choose from.
Version 0.2.0 - Updated with SEARCH_WEB and 7-action space.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import IntEnum


class ActionType(IntEnum):
    """Enumeration of action types matching DQN output."""
    DIRECT_RESPONSE = 0
    CLARIFY_QUESTION = 1
    USE_TOOL = 2
    SEARCH_WEB = 3
    BREAK_DOWN_TASK = 4
    REFLECT_AND_REASON = 5
    DEFER_OR_DECLINE = 6


@dataclass
class Action:
    """Represents a single action with metadata."""
    id: int
    name: str
    description: str
    requires_tool: bool = False
    requires_approval: bool = False
    requires_search: bool = False
    priority: int = 0  # Higher = preferred when tie-breaking
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"Action({self.id}: {self.name})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "requires_tool": self.requires_tool,
            "requires_approval": self.requires_approval,
            "requires_search": self.requires_search,
            "priority": self.priority,
        }


class ActionSpace:
    """
    Manages the discrete action space for DQN.
    
    Action IDs:
        0: DIRECT_RESPONSE - Give a direct answer
        1: CLARIFY_QUESTION - Ask for clarification
        2: USE_TOOL - Use a registered tool
        3: SEARCH_WEB - Search the internet
        4: BREAK_DOWN_TASK - Break into steps
        5: REFLECT_AND_REASON - Think carefully
        6: DEFER_OR_DECLINE - Politely decline
    """
    
    # Class-level constants for action IDs
    DIRECT_RESPONSE = 0
    CLARIFY_QUESTION = 1
    USE_TOOL = 2
    SEARCH_WEB = 3
    BREAK_DOWN_TASK = 4
    REFLECT_AND_REASON = 5
    DEFER_OR_DECLINE = 6
    
    def __init__(self, config: Optional[Dict] = None):
        self.actions: List[Action] = []
        self._id_to_action: Dict[int, Action] = {}
        self._name_to_action: Dict[str, Action] = {}
        
        if config:
            self._load_from_config(config)
        else:
            self._load_defaults()
    
    def _load_defaults(self):
        """Load default 7-action space."""
        default_actions = [
            Action(
                id=0,
                name="DIRECT_RESPONSE",
                description="Give a direct, helpful answer from knowledge",
                priority=5,
            ),
            Action(
                id=1,
                name="CLARIFY_QUESTION",
                description="Ask the user for clarification or more details",
                priority=3,
            ),
            Action(
                id=2,
                name="USE_TOOL",
                description="Use a registered tool (calculator, formatter, etc.)",
                requires_tool=True,
                priority=4,
            ),
            Action(
                id=3,
                name="SEARCH_WEB",
                description="Search the internet for current information",
                requires_search=True,
                priority=6,  # High priority for factual queries
            ),
            Action(
                id=4,
                name="BREAK_DOWN_TASK",
                description="Break a complex task into manageable steps",
                priority=4,
            ),
            Action(
                id=5,
                name="REFLECT_AND_REASON",
                description="Think through carefully before responding",
                priority=3,
            ),
            Action(
                id=6,
                name="DEFER_OR_DECLINE",
                description="Politely decline or defer the request",
                priority=1,
            ),
        ]
        
        for action in default_actions:
            self.add_action(action)
    
    def _load_from_config(self, config: Dict):
        """Load actions from config dictionary."""
        actions_cfg = config.get("actions", [])
        
        # Handle list of action dicts
        if isinstance(actions_cfg, list):
            for action_cfg in actions_cfg:
                if isinstance(action_cfg, dict):
                    action = Action(
                        id=action_cfg.get("id", len(self.actions)),
                        name=action_cfg.get("name", f"ACTION_{len(self.actions)}"),
                        description=action_cfg.get("description", ""),
                        requires_tool=action_cfg.get("requires_tool", False),
                        requires_approval=action_cfg.get("requires_approval", False),
                        requires_search=action_cfg.get("requires_search", False),
                        priority=action_cfg.get("priority", 0),
                    )
                    self.add_action(action)
                elif isinstance(action_cfg, str):
                    # Handle simple string list like ["DIRECT_RESPONSE", "CLARIFY_QUESTION"]
                    action = Action(
                        id=len(self.actions),
                        name=action_cfg,
                        description=action_cfg.replace("_", " ").title(),
                    )
                    self.add_action(action)
        
        # If no actions loaded, use defaults
        if not self.actions:
            self._load_defaults()
    
    def add_action(self, action: Action):
        """Register a new action."""
        self.actions.append(action)
        self._id_to_action[action.id] = action
        self._name_to_action[action.name] = action
    
    def get_by_id(self, action_id: int) -> Optional[Action]:
        """Get action by ID."""
        return self._id_to_action.get(action_id)
    
    def get_by_name(self, name: str) -> Optional[Action]:
        """Get action by name."""
        return self._name_to_action.get(name)
    
    @property
    def size(self) -> int:
        """Number of actions in the space."""
        return len(self.actions)
    
    def get_action_names(self) -> List[str]:
        """Get list of all action names."""
        return [a.name for a in self.actions]
    
    def get_tool_actions(self) -> List[Action]:
        """Get actions that require tools."""
        return [a for a in self.actions if a.requires_tool]
    
    def get_search_actions(self) -> List[Action]:
        """Get actions that require web search."""
        return [a for a in self.actions if a.requires_search]
    
    def get_action_prompt(self, action_id: int) -> str:
        """
        Generate behavior guidance for the SLM.
        These prompts guide HOW to respond, not the literal format.
        """
        action = self.get_by_id(action_id)
        if not action:
            return ""
        
        # IMPORTANT: These prompts guide behavior, NOT output format
        # The SLM should still respond naturally, not with numbered lists
        prompts = {
            "DIRECT_RESPONSE": """
BEHAVIOR: Give a direct, helpful answer.
- Respond naturally and conversationally
- Be clear and informative
- Get to the point without unnecessary preamble
- Use your knowledge to give a complete answer
- For simple greetings, just greet back warmly
- Example: "Hello! Great to hear from you. How can I help today?"
DO NOT output numbered lists for simple conversations. Just respond naturally.""",

            "CLARIFY_QUESTION": """
BEHAVIOR: You need more information to help effectively.
- Respond naturally in conversation
- Ask 1-2 focused questions to understand what the user needs
- Be warm and curious, not interrogative
- Show you understood part of what they said
- Example tone: "I'd love to help! Could you tell me more about..."
DO NOT output numbered lists or steps. Respond conversationally.""",

            "USE_TOOL": """
BEHAVIOR: A tool or external capability would help here.
- Identify what tool might help (calculator, formatter, etc.)
- If the tool is available, use it
- If not, explain that tools are being developed
- Offer to help directly instead
Respond naturally about the situation.""",

            "SEARCH_WEB": """
BEHAVIOR: Search the internet for current/accurate information.
- Use web search results to provide accurate, up-to-date information
- Synthesize information from multiple sources
- Cite sources when making specific claims
- Be clear about what you found vs. what you're uncertain about
- If search results are included, use them to answer the question
Respond naturally, incorporating the search findings into your answer.""",

            "BREAK_DOWN_TASK": """
BEHAVIOR: Help the user with a structured approach.
- Organize your response logically
- Break complex topics into manageable parts
- Guide the user through a process step by step
- Use numbered steps ONLY for actual processes with 3+ steps
- For simple queries, still respond naturally without lists
Good for: How-to questions, complex tasks, planning requests.""",

            "REFLECT_AND_REASON": """
BEHAVIOR: Think through this carefully before answering.
- Consider multiple angles and perspectives
- Show your reasoning process naturally
- Be thoughtful and measured in your response
- Good for complex questions, ethical dilemmas, or when uncertain
- Don't overthink simple questions
Respond in natural prose, sharing your thinking conversationally.""",

            "DEFER_OR_DECLINE": """
BEHAVIOR: Politely decline or defer the request.
- Be warm and understanding
- Briefly explain why you can't help with this specific thing
- Offer an alternative if possible
- Keep it short and friendly
- Don't be preachy or lecture the user
Example: "I'm not the best fit for that, but I'd be happy to help with something else!"
Respond naturally and kindly.""",
        }
        
        return prompts.get(action.name, action.description)
    
    def get_action_hints(self, user_message: str) -> Dict[int, float]:
        """
        Get hint scores for each action based on message content.
        Used to guide DQN exploration early in training.
        
        Returns:
            Dict mapping action_id to hint score (0-1)
        """
        message_lower = user_message.lower().strip()
        hints = {i: 0.0 for i in range(self.size)}
        
        # Simple greetings -> DIRECT_RESPONSE
        greeting_words = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 
                         'good evening', 'howdy', 'greetings', 'sup', 'yo']
        if any(word in message_lower for word in greeting_words) and len(message_lower.split()) < 5:
            hints[self.DIRECT_RESPONSE] = 0.9
            return hints
        
        # Feedback/acknowledgment -> DIRECT_RESPONSE
        feedback_words = ['good', 'great', 'thanks', 'thank you', 'ok', 'okay', 
                         'cool', 'nice', 'awesome', 'perfect', 'yes', 'no', 'bye']
        if any(word == message_lower.rstrip('!.') for word in feedback_words):
            hints[self.DIRECT_RESPONSE] = 0.9
            return hints
        
        # Search triggers -> SEARCH_WEB
        search_triggers = [
            'search', 'find', 'look up', 'google', 'what is the latest',
            'current', 'today', 'news', 'recent', 'who is', 'where is',
            'how much', 'price of', 'weather', 'stock', 'score',
        ]
        if any(trigger in message_lower for trigger in search_triggers):
            hints[self.SEARCH_WEB] = 0.8
            hints[self.DIRECT_RESPONSE] = 0.2
        
        # Question words -> might need clarification or direct response
        question_starts = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can you']
        if any(message_lower.startswith(q) for q in question_starts):
            hints[self.DIRECT_RESPONSE] = 0.5
            hints[self.REFLECT_AND_REASON] = 0.3
        
        # Vague or incomplete -> CLARIFY_QUESTION
        vague_indicators = ['help', 'something', 'thing', 'stuff', 'it', 'that']
        if len(message_lower.split()) < 4 and any(v in message_lower for v in vague_indicators):
            hints[self.CLARIFY_QUESTION] = 0.7
        
        # Complex task indicators -> BREAK_DOWN_TASK
        complex_indicators = ['plan', 'steps', 'how do i', 'guide', 'tutorial', 
                            'process', 'explain how', 'walk me through']
        if any(ind in message_lower for ind in complex_indicators):
            hints[self.BREAK_DOWN_TASK] = 0.7
            hints[self.REFLECT_AND_REASON] = 0.2
        
        # Tool triggers -> USE_TOOL
        tool_triggers = ['calculate', 'compute', 'math', 'convert', 'format']
        if any(trigger in message_lower for trigger in tool_triggers):
            hints[self.USE_TOOL] = 0.7
        
        # Sensitive topics -> DEFER_OR_DECLINE
        sensitive_indicators = ['hack', 'illegal', 'harm', 'weapon', 'drug']
        if any(ind in message_lower for ind in sensitive_indicators):
            hints[self.DEFER_OR_DECLINE] = 0.8
        
        return hints
    
    def suggest_action(self, user_message: str) -> int:
        """
        Suggest the best action based on message heuristics.
        Used for bootstrapping before DQN learns.
        """
        hints = self.get_action_hints(user_message)
        
        # Return action with highest hint score
        if max(hints.values()) > 0:
            return max(hints, key=hints.get)
        
        # Default to DIRECT_RESPONSE
        return self.DIRECT_RESPONSE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action space to dictionary."""
        return {
            "size": self.size,
            "actions": [a.to_dict() for a in self.actions],
        }
    
    def __repr__(self):
        return f"ActionSpace({self.size} actions: {', '.join(self.get_action_names())})"


# ===============================
# Quick Test
# ===============================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Action Space")
    print("=" * 60)
    
    space = ActionSpace()
    
    print(f"\n📊 Action Space Info:")
    print(f"   Size: {space.size}")
    print(f"   Actions: {space.get_action_names()}")
    
    print(f"\n🎯 Actions:")
    for action in space.actions:
        flags = []
        if action.requires_tool:
            flags.append("🔧 tool")
        if action.requires_search:
            flags.append("🔍 search")
        flags_str = f" [{', '.join(flags)}]" if flags else ""
        print(f"   {action.id}: {action.name}{flags_str}")
        print(f"      {action.description}")
    
    print(f"\n🔍 Search Actions: {[a.name for a in space.get_search_actions()]}")
    print(f"🔧 Tool Actions: {[a.name for a in space.get_tool_actions()]}")
    
    print(f"\n💡 Action Hint Tests:")
    test_messages = [
        "Hello!",
        "What is the latest version of Python?",
        "Search for AI news",
        "How do I build a web app?",
        "Calculate 25 * 4",
        "hmm",
        "thanks",
        "Can you help me hack something?",
    ]
    
    for msg in test_messages:
        hints = space.get_action_hints(msg)
        best_hint = max(hints, key=hints.get)
        best_action = space.get_by_id(best_hint)
        suggested = space.suggest_action(msg)
        suggested_action = space.get_by_id(suggested)
        
        print(f"\n   '{msg}'")
        print(f"   → Suggested: {suggested_action.name} (hint: {hints[suggested]:.2f})")
        
        # Show top hints
        top_hints = sorted(hints.items(), key=lambda x: x[1], reverse=True)[:3]
        hint_strs = [f"{space.get_by_id(aid).name}:{score:.2f}" for aid, score in top_hints if score > 0]
        if hint_strs:
            print(f"   → Hints: {', '.join(hint_strs)}")
    
    print(f"\n📝 Sample Action Prompt (SEARCH_WEB):")
    prompt = space.get_action_prompt(space.SEARCH_WEB)
    print(prompt[:300] + "...")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
