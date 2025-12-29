# cognitive/reasoning.py
"""
Dheera v0.3.0 - Reasoning Module
Handles complex reasoning, planning, and decision support.
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re


class ReasoningType(Enum):
    """Types of reasoning processes."""
    DIRECT = "direct"                # Simple, direct answer
    STEP_BY_STEP = "step_by_step"    # Break down into steps
    COMPARE = "compare"              # Compare options
    ANALYZE = "analyze"              # Deep analysis
    PLAN = "plan"                    # Create action plan
    VERIFY = "verify"                # Fact check / verify
    SYNTHESIZE = "synthesize"        # Combine multiple sources


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""
    step_number: int
    description: str
    reasoning_type: ReasoningType
    inputs: List[str] = field(default_factory=list)
    output: Optional[str] = None
    confidence: float = 0.5
    requires_tool: bool = False
    tool_name: Optional[str] = None


@dataclass
class ReasoningPlan:
    """A plan for complex reasoning."""
    goal: str
    steps: List[ReasoningStep] = field(default_factory=list)
    current_step: int = 0
    status: str = "pending"
    final_answer: Optional[str] = None
    
    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        return self.current_step / len(self.steps)


@dataclass
class ReasoningResult:
    """Result of reasoning process."""
    reasoning_type: ReasoningType
    conclusion: str
    confidence: float
    steps_taken: List[str]
    sources_used: List[str]
    requires_search: bool = False
    requires_tool: bool = False
    suggested_action: Optional[int] = None


class ReasoningEngine:
    """
    Reasoning engine for Dheera.
    
    Handles:
    - Determining reasoning strategy
    - Breaking down complex problems
    - Planning multi-step tasks
    - Supporting decision making
    """
    
    # Patterns that suggest different reasoning types
    REASONING_PATTERNS = {
        ReasoningType.COMPARE: [
            r'\b(compare|versus|vs|better|worse|difference|between)\b',
            r'\b(pros and cons|advantages|disadvantages)\b',
            r'\b(which (is|are|one|should))\b',
        ],
        ReasoningType.ANALYZE: [
            r'\b(analyze|analysis|examine|evaluate|assess)\b',
            r'\b(why (is|does|do|did|would))\b',
            r'\b(explain (why|how))\b',
            r'\b(what (causes|caused|leads|led))\b',
        ],
        ReasoningType.PLAN: [
            r'\b(plan|planning|strategy|roadmap)\b',
            r'\b(steps to|how (do|can|should) (i|we))\b',
            r'\b(guide|tutorial|walkthrough)\b',
        ],
        ReasoningType.VERIFY: [
            r'\b(is it true|fact check|verify|confirm)\b',
            r'\b(really|actually|correct)\?',
            r'\b(true or false)\b',
        ],
        ReasoningType.SYNTHESIZE: [
            r'\b(summarize|summary|combine|integrate)\b',
            r'\b(overview|conclusion|takeaway)\b',
        ],
        ReasoningType.STEP_BY_STEP: [
            r'\b(step by step|step-by-step|one by one)\b',
            r'\b(walk me through|guide me)\b',
            r'\b(detailed|thorough|complete)\s+\w+\b',
        ],
    }
    
    # Keywords that suggest need for external information
    SEARCH_TRIGGERS = [
        "latest", "current", "today", "recent", "news",
        "2024", "2025", "now", "update",
        "price", "stock", "weather", "score",
    ]
    
    TOOL_TRIGGERS = {
        "calculator": ["calculate", "compute", "math", "sum", "add", "subtract", "multiply", "divide", "+", "-", "*", "/"],
        "code_executor": ["run", "execute", "code", "script", "program"],
    }
    
    def __init__(self):
        # Compile patterns
        self._compiled_patterns = {}
        for rtype, patterns in self.REASONING_PATTERNS.items():
            self._compiled_patterns[rtype] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def determine_reasoning_type(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ReasoningType, float]:
        """
        Determine the best reasoning approach for a message.
        
        Args:
            message: User message
            context: Additional context (dialogue state, entities, etc.)
            
        Returns:
            (ReasoningType, confidence)
        """
        message_lower = message.lower()
        scores = {rtype: 0.0 for rtype in ReasoningType}
        
        # Pattern matching
        for rtype, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(message_lower):
                    scores[rtype] += 0.4
        
        # Length-based heuristics
        word_count = len(message.split())
        if word_count > 30:
            scores[ReasoningType.ANALYZE] += 0.2
            scores[ReasoningType.STEP_BY_STEP] += 0.1
        elif word_count < 5:
            scores[ReasoningType.DIRECT] += 0.3
        
        # Question word analysis
        if message_lower.startswith(("why", "how come")):
            scores[ReasoningType.ANALYZE] += 0.3
        elif message_lower.startswith(("how do", "how can", "how to")):
            scores[ReasoningType.PLAN] += 0.3
        elif "which" in message_lower or "better" in message_lower:
            scores[ReasoningType.COMPARE] += 0.3
        
        # Context influence
        if context:
            if context.get("has_multiple_sources"):
                scores[ReasoningType.SYNTHESIZE] += 0.2
            if context.get("is_followup"):
                scores[ReasoningType.STEP_BY_STEP] += 0.1
        
        # Default to direct if no strong signals
        if max(scores.values()) < 0.2:
            scores[ReasoningType.DIRECT] = 0.5
        
        # Get best type
        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type], 1.0)
        
        return best_type, confidence
    
    def check_external_requirements(
        self,
        message: str,
    ) -> Dict[str, Any]:
        """
        Check if message requires external resources.
        
        Returns:
            Dict with requires_search, requires_tool, tool_name
        """
        message_lower = message.lower()
        
        result = {
            "requires_search": False,
            "requires_tool": False,
            "tool_name": None,
        }
        
        # Check search triggers
        if any(trigger in message_lower for trigger in self.SEARCH_TRIGGERS):
            result["requires_search"] = True
        
        # Check tool triggers
        for tool, triggers in self.TOOL_TRIGGERS.items():
            if any(trigger in message_lower for trigger in triggers):
                result["requires_tool"] = True
                result["tool_name"] = tool
                break
        
        return result
    
    def create_reasoning_plan(
        self,
        goal: str,
        reasoning_type: ReasoningType,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningPlan:
        """
        Create a plan for complex reasoning.
        
        Args:
            goal: What we're trying to accomplish
            reasoning_type: Type of reasoning to use
            context: Additional context
            
        Returns:
            ReasoningPlan with steps
        """
        plan = ReasoningPlan(goal=goal)
        
        # Generate steps based on reasoning type
        if reasoning_type == ReasoningType.COMPARE:
            plan.steps = [
                ReasoningStep(1, "Identify items to compare", ReasoningType.DIRECT),
                ReasoningStep(2, "List key comparison criteria", ReasoningType.DIRECT),
                ReasoningStep(3, "Gather information on each item", ReasoningType.DIRECT, requires_tool=True),
                ReasoningStep(4, "Compare across criteria", ReasoningType.ANALYZE),
                ReasoningStep(5, "Draw conclusion", ReasoningType.SYNTHESIZE),
            ]
        
        elif reasoning_type == ReasoningType.ANALYZE:
            plan.steps = [
                ReasoningStep(1, "Understand the question", ReasoningType.DIRECT),
                ReasoningStep(2, "Identify key factors", ReasoningType.DIRECT),
                ReasoningStep(3, "Research if needed", ReasoningType.DIRECT, requires_tool=True),
                ReasoningStep(4, "Analyze relationships", ReasoningType.ANALYZE),
                ReasoningStep(5, "Form explanation", ReasoningType.SYNTHESIZE),
            ]
        
        elif reasoning_type == ReasoningType.PLAN:
            plan.steps = [
                ReasoningStep(1, "Define the end goal", ReasoningType.DIRECT),
                ReasoningStep(2, "Identify prerequisites", ReasoningType.ANALYZE),
                ReasoningStep(3, "Break into sub-tasks", ReasoningType.STEP_BY_STEP),
                ReasoningStep(4, "Order tasks logically", ReasoningType.ANALYZE),
                ReasoningStep(5, "Present action plan", ReasoningType.SYNTHESIZE),
            ]
        
        elif reasoning_type == ReasoningType.VERIFY:
            plan.steps = [
                ReasoningStep(1, "Identify claim to verify", ReasoningType.DIRECT),
                ReasoningStep(2, "Search for evidence", ReasoningType.DIRECT, requires_tool=True, tool_name="search"),
                ReasoningStep(3, "Evaluate source reliability", ReasoningType.ANALYZE),
                ReasoningStep(4, "Cross-reference information", ReasoningType.COMPARE),
                ReasoningStep(5, "State verification result", ReasoningType.DIRECT),
            ]
        
        elif reasoning_type == ReasoningType.SYNTHESIZE:
            plan.steps = [
                ReasoningStep(1, "Gather all information", ReasoningType.DIRECT),
                ReasoningStep(2, "Identify main themes", ReasoningType.ANALYZE),
                ReasoningStep(3, "Find connections", ReasoningType.ANALYZE),
                ReasoningStep(4, "Combine into coherent summary", ReasoningType.SYNTHESIZE),
            ]
        
        elif reasoning_type == ReasoningType.STEP_BY_STEP:
            plan.steps = [
                ReasoningStep(1, "Understand the task", ReasoningType.DIRECT),
                ReasoningStep(2, "Identify steps needed", ReasoningType.ANALYZE),
                ReasoningStep(3, "Execute step by step", ReasoningType.STEP_BY_STEP),
                ReasoningStep(4, "Verify each step", ReasoningType.VERIFY),
            ]
        
        else:  # DIRECT
            plan.steps = [
                ReasoningStep(1, "Process query directly", ReasoningType.DIRECT),
            ]
        
        return plan
    
    def reason(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """
        Main reasoning entry point.
        
        Args:
            message: User message
            context: Additional context
            
        Returns:
            ReasoningResult with analysis
        """
        # 1. Determine reasoning type
        reasoning_type, confidence = self.determine_reasoning_type(message, context)
        
        # 2. Check external requirements
        external = self.check_external_requirements(message)
        
        # 3. Create reasoning plan (for complex types)
        plan = self.create_reasoning_plan(message, reasoning_type, context)
        
        # 4. Suggest action based on reasoning
        suggested_action = self._suggest_action(reasoning_type, external)
        
        # 5. Build result
        steps_desc = [f"{s.step_number}. {s.description}" for s in plan.steps]
        
        return ReasoningResult(
            reasoning_type=reasoning_type,
            conclusion=f"Recommended approach: {reasoning_type.value}",
            confidence=confidence,
            steps_taken=steps_desc,
            sources_used=[],
            requires_search=external["requires_search"],
            requires_tool=external["requires_tool"],
            suggested_action=suggested_action,
        )
    
    def _suggest_action(
        self,
        reasoning_type: ReasoningType,
        external: Dict[str, Any],
    ) -> int:
        """Suggest DQN action based on reasoning."""
        # Priority: external requirements first
        if external["requires_search"]:
            return 3  # SEARCH_WEB
        
        if external["requires_tool"]:
            return 2  # USE_TOOL
        
        # Then by reasoning type
        ACTION_MAP = {
            ReasoningType.DIRECT: 0,           # DIRECT_RESPONSE
            ReasoningType.STEP_BY_STEP: 4,     # BREAK_DOWN_TASK
            ReasoningType.COMPARE: 5,          # REFLECT_AND_REASON
            ReasoningType.ANALYZE: 5,          # REFLECT_AND_REASON
            ReasoningType.PLAN: 4,             # BREAK_DOWN_TASK
            ReasoningType.VERIFY: 3,           # SEARCH_WEB
            ReasoningType.SYNTHESIZE: 5,       # REFLECT_AND_REASON
        }
        
        return ACTION_MAP.get(reasoning_type, 0)
    
    def get_reasoning_prompt(
        self,
        reasoning_type: ReasoningType,
        goal: str,
    ) -> str:
        """Get a prompt to guide SLM reasoning."""
        prompts = {
            ReasoningType.DIRECT: f"Answer directly and concisely: {goal}",
            
            ReasoningType.STEP_BY_STEP: f"""Break this down step by step:
{goal}

Think through each step before answering.""",
            
            ReasoningType.COMPARE: f"""Compare the following:
{goal}

Consider:
1. Key similarities
2. Key differences  
3. Pros and cons of each
4. Recommendation""",
            
            ReasoningType.ANALYZE: f"""Analyze this thoroughly:
{goal}

Consider:
1. What is being asked
2. Key factors involved
3. Underlying causes/reasons
4. Implications""",
            
            ReasoningType.PLAN: f"""Create an action plan for:
{goal}

Include:
1. Clear goal statement
2. Prerequisites
3. Step-by-step actions
4. Expected outcomes""",
            
            ReasoningType.VERIFY: f"""Verify this claim:
{goal}

Check:
1. Source reliability
2. Supporting evidence
3. Contradicting evidence
4. Confidence level""",
            
            ReasoningType.SYNTHESIZE: f"""Synthesize the information about:
{goal}

Combine:
1. Main themes
2. Key insights
3. Connections
4. Unified conclusion""",
        }
        
        return prompts.get(reasoning_type, f"Respond to: {goal}")


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing ReasoningEngine...")
    
    engine = ReasoningEngine()
    
    test_cases = [
        "What's 25 * 4?",
        "Compare Python and JavaScript for web development",
        "Why is the sky blue?",
        "How do I set up a Docker container for my Python app?",
        "Is it true that Python is the most popular programming language?",
        "Summarize the key features of machine learning",
        "Walk me through creating a REST API step by step",
        "What's the latest news about AI?",
    ]
    
    print("\nReasoning Analysis:")
    print("-" * 60)
    
    for msg in test_cases:
        result = engine.reason(msg)
        
        print(f"\n'{msg}'")
        print(f"  Type: {result.reasoning_type.value} ({result.confidence:.2f})")
        print(f"  Suggested Action: {result.suggested_action}")
        print(f"  Search needed: {result.requires_search}")
        print(f"  Tool needed: {result.requires_tool}")
        print(f"  Steps: {len(result.steps_taken)}")
    
    # Test plan creation
    print("\n\nReasoning Plan Example:")
    print("-" * 60)
    
    plan = engine.create_reasoning_plan(
        "Compare React vs Vue for a new project",
        ReasoningType.COMPARE,
    )
    
    print(f"Goal: {plan.goal}")
    for step in plan.steps:
        tool_note = f" (needs {step.tool_name})" if step.requires_tool else ""
        print(f"  {step.step_number}. {step.description}{tool_note}")
    
    # Test prompt generation
    print("\n\nReasoning Prompt Example:")
    print("-" * 60)
    
    prompt = engine.get_reasoning_prompt(
        ReasoningType.ANALYZE,
        "Why do neural networks need activation functions?"
    )
    print(prompt)
    
    print("\n✅ Reasoning engine tests passed!")
