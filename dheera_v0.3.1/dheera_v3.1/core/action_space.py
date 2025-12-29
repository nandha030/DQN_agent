# core/action_space.py
"""
Dheera v0.3.0 - Action Space
Defines the 8-action decision space for Rainbow DQN.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    """Action IDs for DQN."""
    DIRECT_RESPONSE = 0
    CLARIFY_QUESTION = 1
    USE_TOOL = 2
    SEARCH_WEB = 3
    BREAK_DOWN_TASK = 4
    REFLECT_AND_REASON = 5
    DEFER_OR_DECLINE = 6
    COGNITIVE_PROCESS = 7


@dataclass
class ActionInfo:
    """Information about an action."""
    id: int
    name: str
    description: str
    priority: int  # Lower = higher priority for heuristics
    requires_tool: bool = False
    requires_search: bool = False
    triggers: List[str] = None
    
    def __post_init__(self):
        if self.triggers is None:
            self.triggers = []


class ActionSpace:
    """
    Defines and manages the 8-action space.
    
    Actions:
    0. DIRECT_RESPONSE - Answer directly from knowledge
    1. CLARIFY_QUESTION - Ask for clarification
    2. USE_TOOL - Execute a tool (calculator, code, etc.)
    3. SEARCH_WEB - Search the internet
    4. BREAK_DOWN_TASK - Decompose complex task
    5. REFLECT_AND_REASON - Step-by-step thinking
    6. DEFER_OR_DECLINE - Politely decline
    7. COGNITIVE_PROCESS - Use cognitive layer
    """
    
    NUM_ACTIONS = 8
    
    ACTIONS = {
        Action.DIRECT_RESPONSE: ActionInfo(
            id=0,
            name="DIRECT_RESPONSE",
            description="Answer directly and concisely",
            priority=3,
            triggers=["hello", "hi", "thanks", "what is", "who is", "define"],
        ),
        
        Action.CLARIFY_QUESTION: ActionInfo(
            id=1,
            name="CLARIFY_QUESTION",
            description="Ask for clarification when question is ambiguous",
            priority=4,
            triggers=["what do you mean", "unclear", "ambiguous"],
        ),
        
        Action.USE_TOOL: ActionInfo(
            id=2,
            name="USE_TOOL",
            description="Use a tool like calculator or code executor",
            priority=2,
            requires_tool=True,
            triggers=["calculate", "compute", "run code", "execute"],
        ),
        
        Action.SEARCH_WEB: ActionInfo(
            id=3,
            name="SEARCH_WEB",
            description="Search the internet for information",
            priority=1,
            requires_search=True,
            triggers=[
                "search", "find", "look up", "latest", "current",
                "news", "today", "recent", "price", "weather",
                "who is the current", "what happened",
            ],
        ),
        
        Action.BREAK_DOWN_TASK: ActionInfo(
            id=4,
            name="BREAK_DOWN_TASK",
            description="Break complex task into steps",
            priority=3,
            triggers=["how do i", "how to", "steps to", "guide", "tutorial"],
        ),
        
        Action.REFLECT_AND_REASON: ActionInfo(
            id=5,
            name="REFLECT_AND_REASON",
            description="Think step-by-step for complex questions",
            priority=4,
            triggers=["why", "explain", "analyze", "compare", "think about"],
        ),
        
        Action.DEFER_OR_DECLINE: ActionInfo(
            id=6,
            name="DEFER_OR_DECLINE",
            description="Politely decline inappropriate requests",
            priority=5,
            triggers=["hack", "illegal", "harmful"],
        ),
        
        Action.COGNITIVE_PROCESS: ActionInfo(
            id=7,
            name="COGNITIVE_PROCESS",
            description="Use cognitive layer for complex processing",
            priority=4,
            triggers=["understand", "context", "remember", "follow up"],
        ),
    }
    
    def __init__(self):
        # Build trigger lookup
        self._trigger_map: Dict[str, int] = {}
        for action, info in self.ACTIONS.items():
            for trigger in info.triggers:
                self._trigger_map[trigger.lower()] = action.value
    
    def get_action_info(self, action_id: int) -> ActionInfo:
        """Get information about an action."""
        return self.ACTIONS.get(Action(action_id))
    
    def get_action_name(self, action_id: int) -> str:
        """Get action name."""
        info = self.get_action_info(action_id)
        return info.name if info else f"ACTION_{action_id}"
    
    def get_heuristic_action(self, message: str) -> Optional[int]:
        """
        Get suggested action based on message triggers.
        
        Returns:
            Action ID or None if no clear match
        """
        message_lower = message.lower()
        
        # Check triggers
        matched_actions = []
        for trigger, action_id in self._trigger_map.items():
            if trigger in message_lower:
                info = self.ACTIONS[Action(action_id)]
                matched_actions.append((action_id, info.priority))
        
        if not matched_actions:
            return None
        
        # Return highest priority (lowest number)
        matched_actions.sort(key=lambda x: x[1])
        return matched_actions[0][0]
    
    def should_search(self, message: str) -> bool:
        """Check if message suggests search is needed."""
        search_info = self.ACTIONS[Action.SEARCH_WEB]
        message_lower = message.lower()
        
        return any(trigger in message_lower for trigger in search_info.triggers)
    
    def should_use_tool(self, message: str) -> bool:
        """Check if message suggests tool usage."""
        tool_info = self.ACTIONS[Action.USE_TOOL]
        message_lower = message.lower()
        
        return any(trigger in message_lower for trigger in tool_info.triggers)
    
    def get_action_mask(
        self,
        tools_available: bool = True,
        search_available: bool = True,
    ) -> List[bool]:
        """
        Get mask of available actions.
        
        Returns:
            List of booleans (True = action available)
        """
        mask = [True] * self.NUM_ACTIONS
        
        if not tools_available:
            mask[Action.USE_TOOL] = False
        
        if not search_available:
            mask[Action.SEARCH_WEB] = False
        
        return mask
    
    def get_all_actions(self) -> List[ActionInfo]:
        """Get all action definitions."""
        return list(self.ACTIONS.values())
    
    def get_action_prompt_hint(self, action_id: int) -> str:
        """Get prompt hint for action."""
        hints = {
            0: "Provide a direct, helpful answer.",
            1: "Ask a clarifying question to better understand.",
            2: "Use the tool result to inform your response.",
            3: "Use the search results to provide current information.",
            4: "Break this down into clear, numbered steps.",
            5: "Think through this step by step.",
            6: "Politely explain you cannot help with this.",
            7: "Use the cognitive analysis to inform your response.",
        }
        return hints.get(action_id, "Respond helpfully.")


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing ActionSpace...")
    
    space = ActionSpace()
    
    # Test action info
    for i in range(space.NUM_ACTIONS):
        info = space.get_action_info(i)
        print(f"  {i}: {info.name} - {info.description[:40]}...")
    
    print(f"\n✓ Total actions: {space.NUM_ACTIONS}")
    
    # Test heuristic matching
    test_cases = [
        ("Hello!", Action.DIRECT_RESPONSE),
        ("Search for Python tutorials", Action.SEARCH_WEB),
        ("Calculate 25 * 4", Action.USE_TOOL),
        ("How do I install Docker?", Action.BREAK_DOWN_TASK),
        ("Why is the sky blue?", Action.REFLECT_AND_REASON),
    ]
    
    print("\nHeuristic Action Tests:")
    for msg, expected in test_cases:
        suggested = space.get_heuristic_action(msg)
        match = "✓" if suggested == expected else "✗"
        print(f"  {match} '{msg}' -> {space.get_action_name(suggested or 0)}")
    
    # Test masks
    mask = space.get_action_mask(tools_available=False)
    print(f"\n✓ Action mask (no tools): {mask}")
    
    # Test search/tool detection
    print(f"\n✓ Should search 'latest news': {space.should_search('latest news')}")
    print(f"✓ Should use tool 'calculate 5+5': {space.should_use_tool('calculate 5+5')}")
    
    print("\n✅ Action space tests passed!")
