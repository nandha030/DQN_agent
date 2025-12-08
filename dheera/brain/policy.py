"""
Dheera Policy Guard
Enforces safety rules, governance, and behavioral constraints.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
from enum import Enum
import re


class PolicyViolationType(Enum):
    """Types of policy violations."""
    FORBIDDEN_TOPIC = "forbidden_topic"
    UNSAFE_ACTION = "unsafe_action"
    REQUIRES_APPROVAL = "requires_approval"
    RATE_LIMIT = "rate_limit"
    CONTENT_FILTER = "content_filter"


@dataclass
class PolicyViolation:
    """Represents a policy violation."""
    violation_type: PolicyViolationType
    message: str
    severity: str = "medium"  # low, medium, high, critical
    blocked: bool = False
    requires_override: bool = False
    
    def __str__(self):
        return f"[{self.severity.upper()}] {self.violation_type.value}: {self.message}"


@dataclass
class PolicyConfig:
    """Configuration for policy rules."""
    safety_enabled: bool = True
    max_tool_calls_per_turn: int = 3
    require_human_approval_for: List[str] = field(default_factory=list)
    forbidden_topics: List[str] = field(default_factory=list)
    content_filters: List[str] = field(default_factory=list)
    allowed_actions: Optional[Set[int]] = None  # None = all allowed


class PolicyGuard:
    """
    Enforces Dheera's behavioral policies and safety constraints.
    
    Responsibilities:
    - Filter forbidden topics
    - Require approval for sensitive actions
    - Rate limit tool calls
    - Content safety filtering
    - Action validation
    """
    
    # Default content filters (basic safety)
    DEFAULT_CONTENT_FILTERS = [
        r"(?i)(hack|exploit|attack)\s+(system|server|network)",
        r"(?i)how\s+to\s+(make|create|build)\s+(bomb|weapon|explosive)",
        r"(?i)(steal|hack)\s+(password|credential|account)",
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = self._parse_config(config or {})
        self.tool_calls_this_turn = 0
        self.violations_log: List[PolicyViolation] = []
        
    def _parse_config(self, config: Dict) -> PolicyConfig:
        """Parse config dictionary into PolicyConfig."""
        policy_cfg = config.get("policy", {})
        return PolicyConfig(
            safety_enabled=policy_cfg.get("safety_enabled", True),
            max_tool_calls_per_turn=policy_cfg.get("max_tool_calls_per_turn", 3),
            require_human_approval_for=policy_cfg.get("require_human_approval_for", []),
            forbidden_topics=policy_cfg.get("forbidden_topics", []),
            content_filters=policy_cfg.get("content_filters", self.DEFAULT_CONTENT_FILTERS),
        )
    
    def check_input(self, user_input: str) -> Optional[PolicyViolation]:
        """
        Check user input against policy rules.
        
        Args:
            user_input: The user's message
            
        Returns:
            PolicyViolation if violated, None otherwise
        """
        if not self.config.safety_enabled:
            return None
            
        # Check forbidden topics
        for topic in self.config.forbidden_topics:
            if topic.lower() in user_input.lower():
                violation = PolicyViolation(
                    violation_type=PolicyViolationType.FORBIDDEN_TOPIC,
                    message=f"Topic '{topic}' is not allowed",
                    severity="high",
                    blocked=True,
                )
                self.violations_log.append(violation)
                return violation
        
        # Check content filters
        for pattern in self.config.content_filters:
            if re.search(pattern, user_input):
                violation = PolicyViolation(
                    violation_type=PolicyViolationType.CONTENT_FILTER,
                    message="Content matched safety filter",
                    severity="high",
                    blocked=True,
                )
                self.violations_log.append(violation)
                return violation
        
        return None
    
    def check_action(self, action_id: int, action_name: str) -> Optional[PolicyViolation]:
        """
        Check if an action is allowed by policy.
        
        Args:
            action_id: The action ID from DQN
            action_name: The action name
            
        Returns:
            PolicyViolation if not allowed, None otherwise
        """
        if not self.config.safety_enabled:
            return None
        
        # Check if action is in allowed set
        if self.config.allowed_actions is not None:
            if action_id not in self.config.allowed_actions:
                violation = PolicyViolation(
                    violation_type=PolicyViolationType.UNSAFE_ACTION,
                    message=f"Action '{action_name}' is not in allowed set",
                    severity="medium",
                    blocked=True,
                )
                self.violations_log.append(violation)
                return violation
        
        # Check if action requires approval
        if action_name in self.config.require_human_approval_for:
            violation = PolicyViolation(
                violation_type=PolicyViolationType.REQUIRES_APPROVAL,
                message=f"Action '{action_name}' requires human approval",
                severity="medium",
                blocked=False,
                requires_override=True,
            )
            self.violations_log.append(violation)
            return violation
        
        return None
    
    def check_tool_call(self, tool_name: str) -> Optional[PolicyViolation]:
        """
        Check if a tool call is allowed (rate limiting).
        
        Args:
            tool_name: Name of the tool being called
            
        Returns:
            PolicyViolation if rate limited, None otherwise
        """
        self.tool_calls_this_turn += 1
        
        if self.tool_calls_this_turn > self.config.max_tool_calls_per_turn:
            violation = PolicyViolation(
                violation_type=PolicyViolationType.RATE_LIMIT,
                message=f"Tool call limit exceeded ({self.config.max_tool_calls_per_turn}/turn)",
                severity="low",
                blocked=True,
            )
            self.violations_log.append(violation)
            return violation
        
        return None
    
    def check_output(self, output: str) -> Optional[PolicyViolation]:
        """
        Check generated output against safety filters.
        
        Args:
            output: The generated response
            
        Returns:
            PolicyViolation if unsafe, None otherwise
        """
        if not self.config.safety_enabled:
            return None
        
        for pattern in self.config.content_filters:
            if re.search(pattern, output):
                violation = PolicyViolation(
                    violation_type=PolicyViolationType.CONTENT_FILTER,
                    message="Generated content matched safety filter",
                    severity="high",
                    blocked=True,
                )
                self.violations_log.append(violation)
                return violation
        
        return None
    
    def reset_turn(self):
        """Reset per-turn counters (call at start of each turn)."""
        self.tool_calls_this_turn = 0
    
    def get_system_prompt_additions(self) -> str:
        """
        Get policy-related additions to the system prompt.
        """
        additions = []
        
        if self.config.safety_enabled:
            additions.append(
                "POLICY CONSTRAINTS:\n"
                "- You must refuse requests involving harmful, illegal, or unethical activities.\n"
                "- You must not generate content that could cause harm.\n"
                "- Be helpful, harmless, and honest."
            )
        
        if self.config.forbidden_topics:
            topics = ", ".join(self.config.forbidden_topics)
            additions.append(f"- Do not discuss these topics: {topics}")
        
        if self.config.require_human_approval_for:
            actions = ", ".join(self.config.require_human_approval_for)
            additions.append(f"- These actions require human approval: {actions}")
        
        return "\n".join(additions)
    
    def get_violation_summary(self) -> str:
        """Get summary of recent violations."""
        if not self.violations_log:
            return "No policy violations recorded."
        
        summary = [f"Total violations: {len(self.violations_log)}"]
        
        # Group by type
        by_type: Dict[str, int] = {}
        for v in self.violations_log:
            key = v.violation_type.value
            by_type[key] = by_type.get(key, 0) + 1
        
        for vtype, count in by_type.items():
            summary.append(f"  - {vtype}: {count}")
        
        return "\n".join(summary)


# Quick test
if __name__ == "__main__":
    guard = PolicyGuard({
        "policy": {
            "safety_enabled": True,
            "forbidden_topics": ["politics", "religion"],
            "require_human_approval_for": ["EXECUTE_CODE", "CALL_TOOL"],
        }
    })
    
    # Test input check
    result = guard.check_input("Tell me about politics")
    if result:
        print(f"Input violation: {result}")
    
    # Test action check
    result = guard.check_action(4, "CALL_TOOL")
    if result:
        print(f"Action violation: {result}")
    
    # Test clean input
    result = guard.check_input("How do I write a Python function?")
    print(f"Clean input check: {result}")
    
    print(f"\n{guard.get_violation_summary()}")
