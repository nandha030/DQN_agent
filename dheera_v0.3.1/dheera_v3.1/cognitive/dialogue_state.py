# cognitive/dialogue_state.py
"""
Dheera v0.3.0 - Dialogue State Tracker
Tracks conversation state across turns for coherent multi-turn dialogues.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from cognitive.intent_classifier import Intent, IntentResult
from cognitive.entity_extractor import ExtractionResult, Entity, EntityType


class ConversationPhase(Enum):
    """Phases of a conversation."""
    OPENING = "opening"              # Initial greeting/setup
    INFORMATION_GATHERING = "info"   # Collecting requirements
    TASK_EXECUTION = "execution"     # Performing the task
    CLARIFICATION = "clarification"  # Getting clarification
    FOLLOWUP = "followup"            # Follow-up questions
    CLOSING = "closing"              # Wrapping up


class TaskStatus(Enum):
    """Status of the current task."""
    NONE = "none"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_INPUT = "awaiting_input"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Slot:
    """A slot to be filled in the dialogue."""
    name: str
    value: Optional[Any] = None
    required: bool = False
    filled: bool = False
    source_turn: Optional[int] = None


@dataclass
class DialogueState:
    """
    Complete dialogue state at any point in conversation.
    """
    # Conversation tracking
    turn_count: int = 0
    phase: ConversationPhase = ConversationPhase.OPENING
    
    # Current intent
    current_intent: Optional[Intent] = None
    intent_history: List[Intent] = field(default_factory=list)
    
    # Entity tracking
    entities: Dict[str, List[Entity]] = field(default_factory=dict)
    
    # Task/Goal tracking
    current_task: Optional[str] = None
    task_status: TaskStatus = TaskStatus.NONE
    task_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Slot filling
    slots: Dict[str, Slot] = field(default_factory=dict)
    
    # Context
    topic: Optional[str] = None
    topic_history: List[str] = field(default_factory=list)
    
    # User state estimation
    user_satisfaction: float = 0.5  # 0 = frustrated, 1 = satisfied
    user_engagement: float = 0.5    # 0 = disengaged, 1 = engaged
    
    # Flags
    needs_clarification: bool = False
    waiting_for_confirmation: bool = False
    search_performed: bool = False
    tool_used: bool = False
    
    # Timestamps
    last_user_turn: Optional[datetime] = None
    last_assistant_turn: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "turn_count": self.turn_count,
            "phase": self.phase.value,
            "current_intent": self.current_intent.value if self.current_intent else None,
            "current_task": self.current_task,
            "task_status": self.task_status.value,
            "topic": self.topic,
            "user_satisfaction": self.user_satisfaction,
            "user_engagement": self.user_engagement,
            "needs_clarification": self.needs_clarification,
            "slots_filled": sum(1 for s in self.slots.values() if s.filled),
            "slots_total": len(self.slots),
        }


class DialogueStateTracker:
    """
    Tracks and updates dialogue state across conversation turns.
    """
    
    def __init__(self):
        self.state = DialogueState()
        self.history: List[Dict[str, Any]] = []
    
    def update(
        self,
        user_message: str,
        assistant_response: str,
        intent_result: Optional[IntentResult] = None,
        entity_result: Optional[ExtractionResult] = None,
        action_taken: Optional[int] = None,
        reward: Optional[float] = None,
    ) -> DialogueState:
        """
        Update dialogue state based on new turn.
        
        Args:
            user_message: What the user said
            assistant_response: What the assistant replied
            intent_result: Classified intent
            entity_result: Extracted entities
            action_taken: DQN action ID
            reward: Reward received
            
        Returns:
            Updated DialogueState
        """
        self.state.turn_count += 1
        self.state.last_user_turn = datetime.now()
        
        # 1. Update intent
        if intent_result:
            self.state.current_intent = intent_result.primary_intent
            self.state.intent_history.append(intent_result.primary_intent)
            
            # Track intent patterns
            self._update_from_intent(intent_result)
        
        # 2. Update entities
        if entity_result:
            for entity in entity_result.entities:
                type_name = entity.type.value
                if type_name not in self.state.entities:
                    self.state.entities[type_name] = []
                self.state.entities[type_name].append(entity)
            
            # Update topic from entities
            if entity_result.topics:
                new_topic = entity_result.topics[0]
                if new_topic != self.state.topic:
                    if self.state.topic:
                        self.state.topic_history.append(self.state.topic)
                    self.state.topic = new_topic
        
        # 3. Update conversation phase
        self._update_phase()
        
        # 4. Update user state estimation
        self._update_user_state(user_message, reward)
        
        # 5. Track action effects
        if action_taken is not None:
            self._update_from_action(action_taken)
        
        # 6. Save to history
        self.history.append({
            "turn": self.state.turn_count,
            "state": self.state.to_dict(),
            "user_message": user_message[:100],
            "assistant_response": assistant_response[:100],
        })
        
        return self.state
    
    def _update_from_intent(self, intent_result: IntentResult):
        """Update state based on detected intent."""
        intent = intent_result.primary_intent
        
        # Handle task-related intents
        if intent in [Intent.REQUEST_CREATE, Intent.REQUEST_SEARCH, 
                      Intent.REQUEST_CALCULATE, Intent.REQUEST_EXPLAIN,
                      Intent.QUESTION_HOW_TO]:
            if self.state.task_status in [TaskStatus.NONE, TaskStatus.COMPLETED]:
                self.state.current_task = intent.value
                self.state.task_status = TaskStatus.PENDING
        
        # Handle clarification/correction
        if intent == Intent.CLARIFICATION:
            self.state.needs_clarification = True
        elif intent == Intent.CORRECTION:
            self.state.task_status = TaskStatus.PENDING  # Reset task
        
        # Handle feedback
        if intent == Intent.FEEDBACK_POSITIVE:
            self.state.user_satisfaction = min(1.0, self.state.user_satisfaction + 0.2)
            if self.state.task_status == TaskStatus.IN_PROGRESS:
                self.state.task_status = TaskStatus.COMPLETED
        elif intent == Intent.FEEDBACK_NEGATIVE:
            self.state.user_satisfaction = max(0.0, self.state.user_satisfaction - 0.2)
        
        # Handle farewell
        if intent == Intent.FAREWELL:
            self.state.phase = ConversationPhase.CLOSING
    
    def _update_phase(self):
        """Update conversation phase based on state."""
        turn = self.state.turn_count
        intent = self.state.current_intent
        
        if turn == 1:
            if intent in [Intent.GREETING, Intent.CHITCHAT]:
                self.state.phase = ConversationPhase.OPENING
            else:
                self.state.phase = ConversationPhase.INFORMATION_GATHERING
        
        elif self.state.needs_clarification:
            self.state.phase = ConversationPhase.CLARIFICATION
        
        elif self.state.task_status == TaskStatus.IN_PROGRESS:
            self.state.phase = ConversationPhase.TASK_EXECUTION
        
        elif intent == Intent.FOLLOWUP:
            self.state.phase = ConversationPhase.FOLLOWUP
        
        elif self.state.task_status == TaskStatus.COMPLETED:
            self.state.phase = ConversationPhase.FOLLOWUP
    
    def _update_user_state(self, user_message: str, reward: Optional[float]):
        """Estimate user satisfaction and engagement."""
        message_lower = user_message.lower()
        
        # Update based on message content
        positive_signals = ["thanks", "great", "good", "perfect", "awesome", "helpful"]
        negative_signals = ["no", "wrong", "not", "don't", "can't", "confused", "frustrated"]
        
        if any(sig in message_lower for sig in positive_signals):
            self.state.user_satisfaction = min(1.0, self.state.user_satisfaction + 0.1)
            self.state.user_engagement = min(1.0, self.state.user_engagement + 0.1)
        
        if any(sig in message_lower for sig in negative_signals):
            self.state.user_satisfaction = max(0.0, self.state.user_satisfaction - 0.1)
        
        # Update from reward
        if reward is not None:
            if reward > 0:
                self.state.user_satisfaction = min(1.0, self.state.user_satisfaction + reward * 0.2)
            elif reward < 0:
                self.state.user_satisfaction = max(0.0, self.state.user_satisfaction + reward * 0.2)
        
        # Engagement from message length
        if len(user_message.split()) > 10:
            self.state.user_engagement = min(1.0, self.state.user_engagement + 0.05)
        elif len(user_message.split()) < 3:
            self.state.user_engagement = max(0.3, self.state.user_engagement - 0.05)
    
    def _update_from_action(self, action: int):
        """Update state based on action taken."""
        # Action IDs: 0=direct, 1=clarify, 2=tool, 3=search, 4=breakdown, 5=reflect, 6=defer, 7=cognitive
        
        if action == 1:  # Clarify
            self.state.waiting_for_confirmation = True
        elif action == 2:  # Tool
            self.state.tool_used = True
        elif action == 3:  # Search
            self.state.search_performed = True
        
        # Update task status
        if self.state.task_status == TaskStatus.PENDING:
            self.state.task_status = TaskStatus.IN_PROGRESS
    
    def add_slot(self, name: str, required: bool = False):
        """Add a slot to track."""
        self.state.slots[name] = Slot(name=name, required=required)
    
    def fill_slot(self, name: str, value: Any):
        """Fill a slot with a value."""
        if name in self.state.slots:
            self.state.slots[name].value = value
            self.state.slots[name].filled = True
            self.state.slots[name].source_turn = self.state.turn_count
    
    def get_unfilled_slots(self) -> List[str]:
        """Get list of unfilled required slots."""
        return [
            name for name, slot in self.state.slots.items()
            if slot.required and not slot.filled
        ]
    
    def get_state_features(self) -> Dict[str, float]:
        """Get state features for DQN state building."""
        return {
            "turn_count_normalized": min(self.state.turn_count / 20.0, 1.0),
            "phase_opening": float(self.state.phase == ConversationPhase.OPENING),
            "phase_execution": float(self.state.phase == ConversationPhase.TASK_EXECUTION),
            "phase_clarification": float(self.state.phase == ConversationPhase.CLARIFICATION),
            "task_pending": float(self.state.task_status == TaskStatus.PENDING),
            "task_in_progress": float(self.state.task_status == TaskStatus.IN_PROGRESS),
            "user_satisfaction": self.state.user_satisfaction,
            "user_engagement": self.state.user_engagement,
            "needs_clarification": float(self.state.needs_clarification),
            "has_topic": float(self.state.topic is not None),
            "entity_count_normalized": min(len(self.state.entities) / 10.0, 1.0),
        }
    
    def reset(self):
        """Reset dialogue state for new conversation."""
        old_state = self.state
        self.state = DialogueState()
        self.history = []
        return old_state
    
    def get_context_summary(self) -> str:
        """Get human-readable context summary for SLM."""
        parts = []
        
        if self.state.topic:
            parts.append(f"Topic: {self.state.topic}")
        
        if self.state.current_task:
            parts.append(f"Task: {self.state.current_task} ({self.state.task_status.value})")
        
        if self.state.entities:
            entity_summary = []
            for etype, ents in self.state.entities.items():
                if ents:
                    entity_summary.append(f"{etype}: {', '.join(e.text for e in ents[:3])}")
            if entity_summary:
                parts.append(f"Entities: {'; '.join(entity_summary)}")
        
        unfilled = self.get_unfilled_slots()
        if unfilled:
            parts.append(f"Need info: {', '.join(unfilled)}")
        
        if parts:
            return " | ".join(parts)
        return ""


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing DialogueStateTracker...")
    
    from cognitive.intent_classifier import IntentClassifier
    from cognitive.entity_extractor import EntityExtractor
    
    tracker = DialogueStateTracker()
    intent_classifier = IntentClassifier()
    entity_extractor = EntityExtractor()
    
    # Simulate a conversation
    conversation = [
        ("Hello!", "Hi there! How can I help you today?"),
        ("Can you help me write a Python script?", "Of course! What should the script do?"),
        ("I want to read a CSV file and filter rows", "I can help with that. What's the filter condition?"),
        ("Filter rows where the 'status' column is 'active'", "Here's a Python script that does that..."),
        ("Thanks, that's perfect!", "You're welcome! Anything else?"),
    ]
    
    print("\nDialogue State Tracking:")
    print("-" * 60)
    
    for user_msg, asst_msg in conversation:
        # Classify and extract
        intent_result = intent_classifier.classify(user_msg)
        entity_result = entity_extractor.extract(user_msg)
        
        # Update state
        state = tracker.update(
            user_message=user_msg,
            assistant_response=asst_msg,
            intent_result=intent_result,
            entity_result=entity_result,
        )
        
        print(f"\nTurn {state.turn_count}:")
        print(f"  User: '{user_msg}'")
        print(f"  Intent: {state.current_intent.value if state.current_intent else 'none'}")
        print(f"  Phase: {state.phase.value}")
        print(f"  Task: {state.current_task} ({state.task_status.value})")
        print(f"  Satisfaction: {state.user_satisfaction:.2f}")
    
    print(f"\nContext Summary: {tracker.get_context_summary()}")
    print(f"\nState Features: {tracker.get_state_features()}")
    
    print("\n✅ Dialogue state tracker tests passed!")
