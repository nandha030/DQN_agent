"""
Dheera Episodic Memory
Long-term memory for storing conversation episodes and learnings.
"""

import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import numpy as np


@dataclass
class MemoryEntry:
    """A single memory entry (one turn in a conversation)."""
    turn_id: int
    user_message: str
    assistant_response: str
    action_taken: int
    action_name: str
    reward: float
    state_vector: List[float]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(**data)


@dataclass
class Episode:
    """A complete conversation episode."""
    episode_id: str
    entries: List[MemoryEntry] = field(default_factory=list)
    total_reward: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    summary: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_entry(self, entry: MemoryEntry):
        """Add a turn to the episode."""
        self.entries.append(entry)
        self.total_reward += entry.reward
    
    def close(self, summary: str = ""):
        """Mark episode as complete."""
        self.end_time = time.time()
        self.summary = summary
    
    @property
    def duration(self) -> float:
        """Episode duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def turn_count(self) -> int:
        return len(self.entries)
    
    @property
    def average_reward(self) -> float:
        if not self.entries:
            return 0.0
        return self.total_reward / len(self.entries)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "entries": [e.to_dict() for e in self.entries],
            "total_reward": self.total_reward,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "summary": self.summary,
            "tags": self.tags,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        entries = [MemoryEntry.from_dict(e) for e in data.pop("entries", [])]
        episode = cls(**data)
        episode.entries = entries
        return episode


class EpisodicMemory:
    """
    Long-term episodic memory for Dheera.
    
    Features:
    - Store complete conversation episodes
    - Search/retrieve relevant past experiences
    - Persist to disk
    - Extract patterns and insights
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_episodes: int = 1000,
        auto_save: bool = True,
    ):
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_episodes = max_episodes
        self.auto_save = auto_save
        
        self.episodes: Dict[str, Episode] = {}
        self.current_episode: Optional[Episode] = None
        self.episode_index: List[str] = []  # Order by time
        
        # Simple embedding cache for search
        self._message_hashes: Dict[str, List[str]] = {}  # hash -> episode_ids
        
        # Load existing memories
        if self.storage_path and self.storage_path.exists():
            self._load()
    
    def _generate_episode_id(self) -> str:
        """Generate unique episode ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"ep_{timestamp}_{random_suffix}"
    
    def start_episode(self, metadata: Optional[Dict] = None) -> Episode:
        """Start a new conversation episode."""
        if self.current_episode:
            self.end_episode()
        
        episode_id = self._generate_episode_id()
        self.current_episode = Episode(
            episode_id=episode_id,
            metadata=metadata or {}
        )
        return self.current_episode
    
    def add_turn(
        self,
        user_message: str,
        assistant_response: str,
        action_taken: int,
        action_name: str,
        reward: float,
        state_vector: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> MemoryEntry:
        """Add a turn to the current episode."""
        if not self.current_episode:
            self.start_episode()
        
        entry = MemoryEntry(
            turn_id=self.current_episode.turn_count,
            user_message=user_message,
            assistant_response=assistant_response,
            action_taken=action_taken,
            action_name=action_name,
            reward=reward,
            state_vector=state_vector.tolist() if hasattr(state_vector, 'tolist') else state_vector,
            metadata=metadata or {},
        )
        
        self.current_episode.add_entry(entry)
        
        # Index for search
        self._index_message(user_message, self.current_episode.episode_id)
        
        return entry
    
    def end_episode(self, summary: str = "", tags: Optional[List[str]] = None):
        """End the current episode and save it."""
        if not self.current_episode:
            return
        
        self.current_episode.close(summary)
        if tags:
            self.current_episode.tags = tags
        
        # Store episode
        self.episodes[self.current_episode.episode_id] = self.current_episode
        self.episode_index.append(self.current_episode.episode_id)
        
        # Enforce max episodes
        while len(self.episode_index) > self.max_episodes:
            oldest_id = self.episode_index.pop(0)
            del self.episodes[oldest_id]
        
        # Auto-save
        if self.auto_save and self.storage_path:
            self._save()
        
        self.current_episode = None
    
    def _index_message(self, message: str, episode_id: str):
        """Index a message for simple search."""
        # Simple word-based indexing
        words = message.lower().split()
        for word in words:
            if len(word) > 3:  # Skip short words
                word_hash = hashlib.md5(word.encode()).hexdigest()[:8]
                if word_hash not in self._message_hashes:
                    self._message_hashes[word_hash] = []
                if episode_id not in self._message_hashes[word_hash]:
                    self._message_hashes[word_hash].append(episode_id)
    
    def search(
        self,
        query: str,
        limit: int = 5,
        min_reward: Optional[float] = None,
    ) -> List[Tuple[Episode, float]]:
        """
        Search for relevant episodes.
        
        Args:
            query: Search query
            limit: Max results
            min_reward: Filter by minimum average reward
            
        Returns:
            List of (Episode, relevance_score) tuples
        """
        # Simple keyword matching
        query_words = query.lower().split()
        episode_scores: Dict[str, int] = {}
        
        for word in query_words:
            if len(word) > 3:
                word_hash = hashlib.md5(word.encode()).hexdigest()[:8]
                if word_hash in self._message_hashes:
                    for ep_id in self._message_hashes[word_hash]:
                        episode_scores[ep_id] = episode_scores.get(ep_id, 0) + 1
        
        # Sort by score
        sorted_eps = sorted(episode_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for ep_id, score in sorted_eps[:limit]:
            episode = self.episodes.get(ep_id)
            if episode:
                if min_reward is None or episode.average_reward >= min_reward:
                    results.append((episode, score / len(query_words)))
        
        return results
    
    def get_recent_episodes(self, n: int = 10) -> List[Episode]:
        """Get most recent episodes."""
        recent_ids = self.episode_index[-n:]
        return [self.episodes[ep_id] for ep_id in reversed(recent_ids) if ep_id in self.episodes]
    
    def get_high_reward_episodes(self, n: int = 10, min_turns: int = 3) -> List[Episode]:
        """Get episodes with highest average reward."""
        eligible = [
            ep for ep in self.episodes.values()
            if ep.turn_count >= min_turns
        ]
        sorted_eps = sorted(eligible, key=lambda x: x.average_reward, reverse=True)
        return sorted_eps[:n]
    
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get a specific episode by ID."""
        return self.episodes.get(episode_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.episodes:
            return {"total_episodes": 0}
        
        rewards = [ep.total_reward for ep in self.episodes.values()]
        turns = [ep.turn_count for ep in self.episodes.values()]
        
        return {
            "total_episodes": len(self.episodes),
            "total_turns": sum(turns),
            "avg_turns_per_episode": np.mean(turns),
            "avg_reward": np.mean(rewards),
            "reward_std": np.std(rewards),
            "best_episode_reward": max(rewards),
            "worst_episode_reward": min(rewards),
        }
    
    def _save(self):
        """Save memory to disk."""
        if not self.storage_path:
            return
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Save episodes
        episodes_file = self.storage_path / "episodes.json"
        data = {
            "episode_index": self.episode_index,
            "episodes": {ep_id: ep.to_dict() for ep_id, ep in self.episodes.items()},
        }
        
        with open(episodes_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self):
        """Load memory from disk."""
        if not self.storage_path:
            return
        
        episodes_file = self.storage_path / "episodes.json"
        if not episodes_file.exists():
            return
        
        try:
            with open(episodes_file, 'r') as f:
                data = json.load(f)
            
            self.episode_index = data.get("episode_index", [])
            
            for ep_id, ep_data in data.get("episodes", {}).items():
                episode = Episode.from_dict(ep_data)
                self.episodes[ep_id] = episode
                
                # Rebuild index
                for entry in episode.entries:
                    self._index_message(entry.user_message, ep_id)
                    
        except Exception as e:
            print(f"Warning: Failed to load episodic memory: {e}")
    
    def export_for_training(self) -> List[Dict[str, Any]]:
        """Export episodes in format suitable for offline training."""
        training_data = []
        
        for episode in self.episodes.values():
            for i, entry in enumerate(episode.entries):
                # Get next state if available
                next_state = (
                    episode.entries[i + 1].state_vector
                    if i + 1 < len(episode.entries)
                    else entry.state_vector
                )
                done = (i + 1 >= len(episode.entries))
                
                training_data.append({
                    "state": entry.state_vector,
                    "action": entry.action_taken,
                    "reward": entry.reward,
                    "next_state": next_state,
                    "done": done,
                    "episode_id": episode.episode_id,
                })
        
        return training_data


# Quick test
if __name__ == "__main__":
    memory = EpisodicMemory(storage_path="./test_memory", auto_save=False)
    
    # Create a test episode
    memory.start_episode(metadata={"test": True})
    
    for i in range(5):
        memory.add_turn(
            user_message=f"Test message {i}",
            assistant_response=f"Test response {i}",
            action_taken=i % 6,
            action_name=f"ACTION_{i % 6}",
            reward=0.5 + (i * 0.1),
            state_vector=np.random.randn(16),
        )
    
    memory.end_episode(summary="Test episode", tags=["test", "demo"])
    
    print(f"Memory stats: {memory.get_stats()}")
    
    # Test search
    results = memory.search("test message")
    print(f"Search results: {len(results)} episodes found")
    
    # Test export
    training_data = memory.export_for_training()
    print(f"Exported {len(training_data)} training samples")
