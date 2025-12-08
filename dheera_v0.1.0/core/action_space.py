"""
Dheera Action Space
Defines discrete actions the DQN can choose from.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import IntEnum


class ActionType(IntEnum):
    ASK_CLARIFYING_QUESTION = 0
    PROVIDE_DETAILED_EXPLANATION = 1
    GIVE_CONCISE_SUMMARY = 2
    PROPOSE_STEP_BY_STEP_PLAN = 3
    CALL_TOOL = 4
    REFLECT_AND_REASON = 5


@dataclass
class Action:
    """Represents a single action with metadata."""
    id: int
    name: str
    description: str
    requires_tool: bool = False
    requires_approval: bool = False
    
    def __repr__(self):
        return f"Action({self.id}: {self.name})"


class ActionSpace:
    """Manages the discrete action space for DQN."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.actions: List[Action] = []
        self._id_to_action: Dict[int, Action] = {}
        self._name_to_action: Dict[str, Action] = {}
        
        if config:
            self._load_from_config(config)
        else:
            self._load_defaults()
    
    def _load_defaults(self):
        """Load default action space."""
        default_actions = [
            Action(id=0, name="ASK_CLARIFYING_QUESTION", description="Ask for clarification"),
            Action(id=1, name="PROVIDE_DETAILED_EXPLANATION", description="Give detailed response"),
            Action(id=2, name="GIVE_CONCISE_SUMMARY", description="Give brief response"),
            Action(id=3, name="PROPOSE_STEP_BY_STEP_PLAN", description="Provide structured guidance"),
            Action(id=4, name="CALL_TOOL", description="Use a tool", requires_tool=True),
            Action(id=5, name="REFLECT_AND_REASON", description="Think through carefully"),
        ]
        
        for action in default_actions:
            self.add_action(action)
    
    def _load_from_config(self, config: Dict):
        """Load actions from config dictionary."""
        for action_cfg in config.get("actions", []):
            action = Action(
                id=action_cfg["id"],
                name=action_cfg["name"],
                description=action_cfg.get("description", ""),
                requires_tool=action_cfg.get("requires_tool", False),
                requires_approval=action_cfg.get("requires_approval", False)
            )
            self.add_action(action)
    
    def add_action(self, action: Action):
        """Register a new action."""
        self.actions.append(action)
        self._id_to_action[action.id] = action
        self._name_to_action[action.name] = action
    
    def get_by_id(self, action_id: int) -> Optional[Action]:
        return self._id_to_action.get(action_id)
    
    def get_by_name(self, name: str) -> Optional[Action]:
        return self._name_to_action.get(name)
    
    @property
    def size(self) -> int:
        return len(self.actions)
    
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
            "ASK_CLARIFYING_QUESTION": """
BEHAVIOR: You need more information to help effectively.
- Respond naturally in conversation
- Ask 1-2 focused questions to understand what the user needs
- Be warm and curious, not interrogative
- Example tone: "I'd love to help! Could you tell me more about..."
DO NOT output numbered lists or steps. Respond conversationally.""",

            "PROVIDE_DETAILED_EXPLANATION": """
BEHAVIOR: Provide a thorough, helpful explanation.
- Give comprehensive information with context and examples
- Explain concepts clearly, assuming the user wants depth
- Be informative but engaging, not dry or lecture-like
- Use natural paragraphs, not bullet points unless truly needed
DO NOT output numbered lists. Write in flowing, natural prose.""",

            "GIVE_CONCISE_SUMMARY": """
BEHAVIOR: Be brief and direct.
- Give a short, clear answer in 1-3 sentences
- Get straight to the point
- No unnecessary elaboration
- Perfect for simple greetings, acknowledgments, or quick answers
- Example: "Hello! Great to chat with you. How can I help?"
DO NOT output numbered lists or multiple options. Just respond directly.""",

            "PROPOSE_STEP_BY_STEP_PLAN": """
BEHAVIOR: Help the user with a structured approach.
- Organize your response logically
- Break complex topics into manageable parts
- Guide the user through a process
- Use clear structure but still write naturally
You may use numbered steps ONLY if explaining a process. For greetings or simple queries, just respond naturally.""",

            "CALL_TOOL": """
BEHAVIOR: A tool or external capability would help here.
- Identify what tool might help
- For now, explain that tools are being developed
- Offer to help directly instead
Respond naturally about the situation.""",

            "REFLECT_AND_REASON": """
BEHAVIOR: Think through this carefully before answering.
- Consider multiple angles
- Show your reasoning process naturally
- Be thoughtful and measured
- Good for complex questions or when you're uncertain
Respond in natural prose, sharing your thinking conversationally.""",
        }
        
        return prompts.get(action.name, action.description)
    
    def __repr__(self):
        return f"ActionSpace({self.size} actions)"
