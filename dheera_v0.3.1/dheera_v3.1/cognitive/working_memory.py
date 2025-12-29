# cognitive/working_memory.py
"""
Dheera v0.3.0 - Working Memory
Short-term memory for current task execution and context.
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import json


@dataclass
class MemoryItem:
    """A single item in working memory."""
    key: str
    value: Any
    importance: float = 0.5  # 0-1 scale
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    source: str = "unknown"  # 'user', 'system', 'rag', 'search', 'tool'
    expires_turns: Optional[int] = None  # Auto-expire after N turns
    
    def access(self):
        """Mark item as accessed."""
        self.accessed_at = datetime.now()
        self.access_count += 1


@dataclass
class TaskContext:
    """Context for the current task."""
    task_id: str
    description: str
    status: str = "pending"
    steps: List[str] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)
    current_step_index: int = 0
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class WorkingMemory:
    """
    Working Memory for Dheera.
    
    Manages:
    - Current task context
    - Recent conversation items
    - Retrieved information (RAG/Search)
    - Tool results
    - Important entities
    """
    
    def __init__(
        self,
        max_items: int = 50,
        max_history: int = 10,
        decay_rate: float = 0.95,
    ):
        self.max_items = max_items
        self.max_history = max_history
        self.decay_rate = decay_rate
        
        # Memory storage
        self._memory: Dict[str, MemoryItem] = {}
        
        # Current task
        self.current_task: Optional[TaskContext] = None
        self.task_history: List[TaskContext] = []
        
        # Conversation history (recent turns)
        self.conversation_history: deque = deque(maxlen=max_history)
        
        # Retrieved context (from RAG/Search)
        self.retrieved_context: List[Dict[str, Any]] = []
        
        # Tool results cache
        self.tool_results: Dict[str, Any] = {}
        
        # Turn counter for expiration
        self._turn_count = 0
    
    def store(
        self,
        key: str,
        value: Any,
        importance: float = 0.5,
        source: str = "unknown",
        expires_turns: Optional[int] = None,
    ):
        """
        Store an item in working memory.
        
        Args:
            key: Unique identifier
            value: The value to store
            importance: How important (0-1)
            source: Where it came from
            expires_turns: Auto-expire after N turns (None = don't expire)
        """
        self._memory[key] = MemoryItem(
            key=key,
            value=value,
            importance=importance,
            source=source,
            expires_turns=expires_turns,
        )
        
        # Manage capacity
        if len(self._memory) > self.max_items:
            self._evict_least_important()
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve an item from memory."""
        if key in self._memory:
            item = self._memory[key]
            item.access()
            return item.value
        return None
    
    def retrieve_by_source(self, source: str) -> List[Tuple[str, Any]]:
        """Retrieve all items from a specific source."""
        results = []
        for key, item in self._memory.items():
            if item.source == source:
                item.access()
                results.append((key, item.value))
        return results
    
    def get_important_items(self, threshold: float = 0.7) -> List[Tuple[str, Any]]:
        """Get items above importance threshold."""
        results = []
        for key, item in self._memory.items():
            if item.importance >= threshold:
                results.append((key, item.value))
        return results
    
    def update_importance(self, key: str, delta: float):
        """Adjust importance of an item."""
        if key in self._memory:
            item = self._memory[key]
            item.importance = max(0.0, min(1.0, item.importance + delta))
    
    def forget(self, key: str):
        """Remove an item from memory."""
        if key in self._memory:
            del self._memory[key]
    
    def _evict_least_important(self):
        """Remove least important items to make room."""
        if not self._memory:
            return
        
        # Sort by importance (ascending)
        sorted_items = sorted(
            self._memory.items(),
            key=lambda x: (x[1].importance, x[1].access_count)
        )
        
        # Remove bottom 20%
        num_to_remove = max(1, len(sorted_items) // 5)
        for key, _ in sorted_items[:num_to_remove]:
            del self._memory[key]
    
    def decay_importance(self):
        """Decay importance of all items over time."""
        for item in self._memory.values():
            item.importance *= self.decay_rate
    
    def new_turn(self):
        """Called at each conversation turn."""
        self._turn_count += 1
        
        # Expire old items
        expired_keys = []
        for key, item in self._memory.items():
            if item.expires_turns is not None:
                item.expires_turns -= 1
                if item.expires_turns <= 0:
                    expired_keys.append(key)
        
        for key in expired_keys:
            del self._memory[key]
        
        # Decay importance
        self.decay_importance()
    
    # ==================== Task Management ====================
    
    def start_task(
        self,
        task_id: str,
        description: str,
        steps: Optional[List[str]] = None,
    ):
        """Start a new task."""
        # Archive current task if exists
        if self.current_task:
            self.task_history.append(self.current_task)
        
        self.current_task = TaskContext(
            task_id=task_id,
            description=description,
            steps=steps or [],
        )
    
    def complete_step(self, step: Optional[str] = None):
        """Mark current step as complete."""
        if self.current_task:
            if step:
                self.current_task.completed_steps.append(step)
            elif self.current_task.steps:
                idx = self.current_task.current_step_index
                if idx < len(self.current_task.steps):
                    self.current_task.completed_steps.append(
                        self.current_task.steps[idx]
                    )
                    self.current_task.current_step_index += 1
    
    def complete_task(self, outputs: Optional[Dict] = None):
        """Mark current task as complete."""
        if self.current_task:
            self.current_task.status = "completed"
            if outputs:
                self.current_task.outputs = outputs
            self.task_history.append(self.current_task)
            self.current_task = None
    
    def fail_task(self, reason: str = ""):
        """Mark current task as failed."""
        if self.current_task:
            self.current_task.status = "failed"
            self.current_task.outputs["failure_reason"] = reason
            self.task_history.append(self.current_task)
            self.current_task = None
    
    def get_task_progress(self) -> Optional[Dict[str, Any]]:
        """Get current task progress."""
        if not self.current_task:
            return None
        
        task = self.current_task
        total_steps = len(task.steps)
        completed = len(task.completed_steps)
        
        return {
            "task_id": task.task_id,
            "description": task.description,
            "status": task.status,
            "progress": completed / total_steps if total_steps > 0 else 0,
            "current_step": task.steps[task.current_step_index] if task.current_step_index < len(task.steps) else None,
            "remaining_steps": total_steps - completed,
        }
    
    # ==================== Conversation History ====================
    
    def add_turn(
        self,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict] = None,
    ):
        """Add a conversation turn to history."""
        self.conversation_history.append({
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        })
    
    def get_recent_context(self, num_turns: int = 3) -> str:
        """Get recent conversation as context string."""
        recent = list(self.conversation_history)[-num_turns:]
        
        parts = []
        for turn in recent:
            parts.append(f"User: {turn['user']}")
            parts.append(f"Assistant: {turn['assistant']}")
        
        return "\n".join(parts)
    
    # ==================== Retrieved Context ====================
    
    def set_retrieved_context(self, context: List[Dict[str, Any]]):
        """Set RAG/search retrieved context."""
        self.retrieved_context = context
        
        # Also store important items in memory
        for i, item in enumerate(context[:5]):  # Top 5
            self.store(
                key=f"retrieved_{i}",
                value=item,
                importance=0.6 - (i * 0.1),  # Decreasing importance
                source="rag",
                expires_turns=5,  # Expire after 5 turns
            )
    
    def get_retrieved_context(self) -> List[Dict[str, Any]]:
        """Get retrieved context."""
        return self.retrieved_context
    
    # ==================== Tool Results ====================
    
    def store_tool_result(self, tool_name: str, result: Any):
        """Store a tool execution result."""
        self.tool_results[tool_name] = {
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Also store in main memory
        self.store(
            key=f"tool_{tool_name}",
            value=result,
            importance=0.7,
            source="tool",
            expires_turns=3,
        )
    
    def get_tool_result(self, tool_name: str) -> Optional[Any]:
        """Get a tool result."""
        if tool_name in self.tool_results:
            return self.tool_results[tool_name]["result"]
        return None
    
    # ==================== Context Building ====================
    
    def build_context_for_slm(self, max_tokens: int = 500) -> str:
        """
        Build context string for SLM prompt.
        
        Includes:
        - Current task info
        - Important memory items
        - Retrieved context
        - Recent tool results
        """
        parts = []
        current_tokens = 0  # Rough estimate
        
        # 1. Current task
        if self.current_task:
            task_info = f"Current Task: {self.current_task.description}"
            if self.current_task.steps:
                progress = self.get_task_progress()
                task_info += f" (Progress: {progress['progress']:.0%})"
            parts.append(task_info)
            current_tokens += len(task_info.split())
        
        # 2. Important memory items
        important = self.get_important_items(threshold=0.6)
        if important:
            items_str = ", ".join(f"{k}={v}" for k, v in important[:3])
            memory_info = f"Context: {items_str}"
            parts.append(memory_info)
            current_tokens += len(memory_info.split())
        
        # 3. Retrieved context (abbreviated)
        if self.retrieved_context:
            rag_info = f"Retrieved: {len(self.retrieved_context)} relevant items"
            parts.append(rag_info)
        
        # 4. Recent tool results
        if self.tool_results:
            tool_names = list(self.tool_results.keys())[:2]
            tools_info = f"Recent tools: {', '.join(tool_names)}"
            parts.append(tools_info)
        
        return " | ".join(parts) if parts else ""
    
    def get_state_features(self) -> Dict[str, float]:
        """Get features for DQN state builder."""
        return {
            "memory_item_count": min(len(self._memory) / self.max_items, 1.0),
            "has_active_task": float(self.current_task is not None),
            "task_progress": self.get_task_progress()["progress"] if self.current_task else 0.0,
            "has_retrieved_context": float(len(self.retrieved_context) > 0),
            "retrieved_count": min(len(self.retrieved_context) / 10, 1.0),
            "conversation_length": min(len(self.conversation_history) / self.max_history, 1.0),
            "has_tool_results": float(len(self.tool_results) > 0),
            "avg_importance": sum(i.importance for i in self._memory.values()) / max(len(self._memory), 1),
        }
    
    def clear(self):
        """Clear all working memory."""
        self._memory.clear()
        self.current_task = None
        self.conversation_history.clear()
        self.retrieved_context.clear()
        self.tool_results.clear()
        self._turn_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "item_count": len(self._memory),
            "turn_count": self._turn_count,
            "has_task": self.current_task is not None,
            "task_history_count": len(self.task_history),
            "conversation_turns": len(self.conversation_history),
            "retrieved_items": len(self.retrieved_context),
            "tool_results": len(self.tool_results),
        }


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing WorkingMemory...")
    
    memory = WorkingMemory()
    
    # Test basic storage
    memory.store("user_name", "Nandha", importance=0.9, source="user")
    memory.store("topic", "Python scripting", importance=0.7, source="system")
    
    print(f"✓ Stored items: {memory.get_stats()['item_count']}")
    print(f"✓ Retrieved user_name: {memory.retrieve('user_name')}")
    
    # Test task management
    memory.start_task(
        task_id="task_1",
        description="Write a CSV parser",
        steps=["Read file", "Parse rows", "Filter data", "Return results"],
    )
    
    print(f"✓ Started task: {memory.get_task_progress()}")
    
    memory.complete_step()
    memory.complete_step()
    
    print(f"✓ After 2 steps: {memory.get_task_progress()}")
    
    # Test conversation history
    memory.add_turn(
        "Help me with Python",
        "Of course! What do you need?",
    )
    
    print(f"✓ Recent context: {memory.get_recent_context(1)}")
    
    # Test retrieved context
    memory.set_retrieved_context([
        {"text": "Python is great", "score": 0.9},
        {"text": "CSV handling in Python", "score": 0.8},
    ])
    
    print(f"✓ Retrieved context: {len(memory.get_retrieved_context())} items")
    
    # Test tool results
    memory.store_tool_result("calculator", {"result": 42})
    print(f"✓ Tool result: {memory.get_tool_result('calculator')}")
    
    # Test context building
    context = memory.build_context_for_slm()
    print(f"✓ SLM Context: {context}")
    
    # Test state features
    features = memory.get_state_features()
    print(f"✓ State features: {features}")
    
    # Test turn decay
    memory.new_turn()
    print(f"✓ After turn decay: {memory.get_stats()}")
    
    print("\n✅ Working memory tests passed!")
