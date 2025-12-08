# memory/episodic_memory.py
"""
Dheera Episodic Memory
Long-term memory for storing conversation episodes and learnings.
Version 0.2.0 - Enhanced with search tracking and 7-action support.
"""

import json
import time
import hashlib
import re
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple, Set
from datetime import datetime
import numpy as np


# Action names (must match action_space.py)
ACTION_NAMES = [
    "DIRECT_RESPONSE",
    "CLARIFY_QUESTION",
    "USE_TOOL",
    "SEARCH_WEB",
    "BREAK_DOWN_TASK",
    "REFLECT_AND_REASON",
    "DEFER_OR_DECLINE",
]


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
    
    # New fields for enhanced tracking
    search_performed: bool = False
    search_query: Optional[str] = None
    search_result_count: int = 0
    tool_used: Optional[str] = None
    response_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        # Handle legacy data without new fields
        defaults = {
            "search_performed": False,
            "search_query": None,
            "search_result_count": 0,
            "tool_used": None,
            "response_latency_ms": 0.0,
        }
        for key, default_val in defaults.items():
            if key not in data:
                data[key] = default_val
        return cls(**data)
    
    @property
    def is_search_action(self) -> bool:
        return self.action_taken == 3 or self.search_performed
    
    @property
    def is_tool_action(self) -> bool:
        return self.action_taken == 2 or self.tool_used is not None


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
    
    # New fields
    action_distribution: Dict[int, int] = field(default_factory=dict)
    search_count: int = 0
    tool_count: int = 0
    
    def add_entry(self, entry: MemoryEntry):
        """Add a turn to the episode."""
        self.entries.append(entry)
        self.total_reward += entry.reward
        
        # Track action distribution
        action_id = entry.action_taken
        self.action_distribution[action_id] = self.action_distribution.get(action_id, 0) + 1
        
        # Track search and tool usage
        if entry.search_performed:
            self.search_count += 1
        if entry.tool_used:
            self.tool_count += 1
    
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
    
    @property
    def search_ratio(self) -> float:
        """Ratio of turns that used search."""
        if not self.entries:
            return 0.0
        return self.search_count / len(self.entries)
    
    @property
    def dominant_action(self) -> Tuple[int, str]:
        """Get the most frequently used action."""
        if not self.action_distribution:
            return (0, "DIRECT_RESPONSE")
        
        action_id = max(self.action_distribution, key=self.action_distribution.get)
        action_name = ACTION_NAMES[action_id] if action_id < len(ACTION_NAMES) else f"ACTION_{action_id}"
        return (action_id, action_name)
    
    def get_action_stats(self) -> Dict[str, Any]:
        """Get detailed action statistics for this episode."""
        stats = {
            "total_turns": self.turn_count,
            "action_counts": {},
            "action_percentages": {},
            "search_count": self.search_count,
            "tool_count": self.tool_count,
        }
        
        for action_id, count in self.action_distribution.items():
            name = ACTION_NAMES[action_id] if action_id < len(ACTION_NAMES) else f"ACTION_{action_id}"
            stats["action_counts"][name] = count
            stats["action_percentages"][name] = round(count / max(1, self.turn_count) * 100, 1)
        
        return stats
    
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
            "action_distribution": self.action_distribution,
            "search_count": self.search_count,
            "tool_count": self.tool_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        entries = [MemoryEntry.from_dict(e) for e in data.pop("entries", [])]
        
        # Handle legacy data
        action_distribution = data.pop("action_distribution", {})
        # Convert string keys to int (JSON serializes dict keys as strings)
        action_distribution = {int(k): v for k, v in action_distribution.items()}
        
        search_count = data.pop("search_count", 0)
        tool_count = data.pop("tool_count", 0)
        
        episode = cls(**data)
        episode.entries = entries
        episode.action_distribution = action_distribution
        episode.search_count = search_count
        episode.tool_count = tool_count
        
        return episode


class EpisodicMemory:
    """
    Long-term episodic memory for Dheera.
    
    Features:
    - Store complete conversation episodes
    - Search/retrieve relevant past experiences
    - Track action patterns and success rates
    - Persist to disk
    - Extract patterns and insights for learning
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_episodes: int = 1000,
        auto_save: bool = True,
        auto_save_interval: int = 5,  # Save every N episodes
    ):
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_episodes = max_episodes
        self.auto_save = auto_save
        self.auto_save_interval = auto_save_interval
        
        self.episodes: Dict[str, Episode] = {}
        self.current_episode: Optional[Episode] = None
        self.episode_index: List[str] = []  # Order by time
        
        # Indexing for search
        self._message_hashes: Dict[str, Set[str]] = {}  # hash -> episode_ids
        self._keyword_index: Dict[str, Set[str]] = {}  # keyword -> episode_ids
        
        # Action pattern tracking
        self._action_reward_history: Dict[int, List[float]] = defaultdict(list)
        self._search_query_history: List[Tuple[str, float]] = []  # (query, reward)
        
        # Stats tracking
        self._episodes_since_save = 0
        
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
        search_performed: bool = False,
        search_query: Optional[str] = None,
        search_result_count: int = 0,
        tool_used: Optional[str] = None,
        response_latency_ms: float = 0.0,
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
            state_vector=state_vector.tolist() if hasattr(state_vector, 'tolist') else list(state_vector),
            metadata=metadata or {},
            search_performed=search_performed,
            search_query=search_query,
            search_result_count=search_result_count,
            tool_used=tool_used,
            response_latency_ms=response_latency_ms,
        )
        
        self.current_episode.add_entry(entry)
        
        # Index for search
        self._index_message(user_message, self.current_episode.episode_id)
        
        # Track action rewards
        self._action_reward_history[action_taken].append(reward)
        
        # Track search queries
        if search_performed and search_query:
            self._search_query_history.append((search_query, reward))
        
        return entry
    
    def end_episode(self, summary: str = "", tags: Optional[List[str]] = None):
        """End the current episode and save it."""
        if not self.current_episode:
            return
        
        self.current_episode.close(summary)
        if tags:
            self.current_episode.tags = tags
        
        # Auto-generate tags if none provided
        if not self.current_episode.tags:
            self.current_episode.tags = self._auto_generate_tags(self.current_episode)
        
        # Store episode
        self.episodes[self.current_episode.episode_id] = self.current_episode
        self.episode_index.append(self.current_episode.episode_id)
        
        # Enforce max episodes
        while len(self.episode_index) > self.max_episodes:
            oldest_id = self.episode_index.pop(0)
            self._remove_from_index(oldest_id)
            del self.episodes[oldest_id]
        
        # Auto-save
        self._episodes_since_save += 1
        if self.auto_save and self.storage_path and self._episodes_since_save >= self.auto_save_interval:
            self._save()
            self._episodes_since_save = 0
        
        self.current_episode = None
    
    def _auto_generate_tags(self, episode: Episode) -> List[str]:
        """Auto-generate tags based on episode content."""
        tags = []
        
        # Tag by dominant action
        action_id, action_name = episode.dominant_action
        tags.append(f"action:{action_name.lower()}")
        
        # Tag by reward level
        avg_reward = episode.average_reward
        if avg_reward > 0.7:
            tags.append("high_reward")
        elif avg_reward < 0.3:
            tags.append("low_reward")
        
        # Tag if search-heavy
        if episode.search_ratio > 0.3:
            tags.append("search_heavy")
        
        # Tag by conversation length
        if episode.turn_count > 10:
            tags.append("long_conversation")
        elif episode.turn_count <= 2:
            tags.append("short_conversation")
        
        # Tag by topics (simple keyword detection)
        all_text = " ".join(e.user_message.lower() for e in episode.entries)
        topic_keywords = {
            "code": ["code", "python", "function", "programming"],
            "math": ["calculate", "math", "formula"],
            "search": ["search", "find", "look up", "what is"],
            "help": ["help", "how to", "explain"],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in all_text for kw in keywords):
                tags.append(f"topic:{topic}")
        
        return tags
    
    def _index_message(self, message: str, episode_id: str):
        """Index a message for search."""
        # Clean and tokenize
        words = re.findall(r'\b\w{3,}\b', message.lower())
        
        for word in set(words):  # Use set to avoid duplicate indexing
            # Word hash for efficient storage
            word_hash = hashlib.md5(word.encode()).hexdigest()[:8]
            
            if word_hash not in self._message_hashes:
                self._message_hashes[word_hash] = set()
            self._message_hashes[word_hash].add(episode_id)
            
            # Also keep keyword index for common terms
            if len(word) >= 4:
                if word not in self._keyword_index:
                    self._keyword_index[word] = set()
                self._keyword_index[word].add(episode_id)
    
    def _remove_from_index(self, episode_id: str):
        """Remove episode from search indices."""
        for hash_set in self._message_hashes.values():
            hash_set.discard(episode_id)
        for keyword_set in self._keyword_index.values():
            keyword_set.discard(episode_id)
    
    def search(
        self,
        query: str,
        limit: int = 5,
        min_reward: Optional[float] = None,
        tags: Optional[List[str]] = None,
        action_filter: Optional[int] = None,
    ) -> List[Tuple[Episode, float]]:
        """
        Search for relevant episodes.
        
        Args:
            query: Search query
            limit: Max results
            min_reward: Filter by minimum average reward
            tags: Filter by tags
            action_filter: Filter by action ID used
            
        Returns:
            List of (Episode, relevance_score) tuples
        """
        # Tokenize query
        query_words = re.findall(r'\b\w{3,}\b', query.lower())
        
        if not query_words:
            return []
        
        # Score episodes
        episode_scores: Dict[str, float] = defaultdict(float)
        
        for word in query_words:
            word_hash = hashlib.md5(word.encode()).hexdigest()[:8]
            
            # Check hash index
            if word_hash in self._message_hashes:
                for ep_id in self._message_hashes[word_hash]:
                    episode_scores[ep_id] += 1.0
            
            # Check keyword index (bonus for exact matches)
            if word in self._keyword_index:
                for ep_id in self._keyword_index[word]:
                    episode_scores[ep_id] += 0.5
        
        # Normalize scores
        max_score = len(query_words) * 1.5
        for ep_id in episode_scores:
            episode_scores[ep_id] /= max_score
        
        # Sort by score
        sorted_eps = sorted(episode_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter and collect results
        results = []
        for ep_id, score in sorted_eps:
            if len(results) >= limit:
                break
            
            episode = self.episodes.get(ep_id)
            if not episode:
                continue
            
            # Apply filters
            if min_reward is not None and episode.average_reward < min_reward:
                continue
            
            if tags and not any(tag in episode.tags for tag in tags):
                continue
            
            if action_filter is not None and action_filter not in episode.action_distribution:
                continue
            
            results.append((episode, score))
        
        return results
    
    def find_similar_situations(
        self,
        state_vector: np.ndarray,
        limit: int = 5,
        distance_threshold: float = 2.0,
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Find similar past situations based on state vector similarity.
        
        Args:
            state_vector: Current state vector
            limit: Max results
            distance_threshold: Max euclidean distance
            
        Returns:
            List of (MemoryEntry, distance) tuples
        """
        state_np = np.array(state_vector)
        similar = []
        
        for episode in self.episodes.values():
            for entry in episode.entries:
                entry_state = np.array(entry.state_vector)
                distance = np.linalg.norm(state_np - entry_state)
                
                if distance <= distance_threshold:
                    similar.append((entry, distance))
        
        # Sort by distance (closest first)
        similar.sort(key=lambda x: x[1])
        
        return similar[:limit]
    
    def get_action_success_rates(self) -> Dict[str, Dict[str, float]]:
        """Get success rates for each action type."""
        stats = {}
        
        for action_id, rewards in self._action_reward_history.items():
            if not rewards:
                continue
            
            name = ACTION_NAMES[action_id] if action_id < len(ACTION_NAMES) else f"ACTION_{action_id}"
            
            rewards_np = np.array(rewards)
            stats[name] = {
                "count": len(rewards),
                "mean_reward": float(np.mean(rewards_np)),
                "std_reward": float(np.std(rewards_np)),
                "success_rate": float(np.mean(rewards_np > 0.5)),  # Reward > 0.5 = success
                "recent_mean": float(np.mean(rewards_np[-50:])) if len(rewards) > 0 else 0.0,
            }
        
        return stats
    
    def get_search_effectiveness(self) -> Dict[str, Any]:
        """Analyze how effective web searches have been."""
        if not self._search_query_history:
            return {"total_searches": 0}
        
        queries, rewards = zip(*self._search_query_history)
        rewards_np = np.array(rewards)
        
        return {
            "total_searches": len(self._search_query_history),
            "mean_reward": float(np.mean(rewards_np)),
            "success_rate": float(np.mean(rewards_np > 0.5)),
            "recent_searches": list(self._search_query_history[-10:]),
        }
    
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
    
    def get_episodes_by_action(self, action_id: int, n: int = 10) -> List[Episode]:
        """Get episodes where a specific action was used."""
        matching = [
            ep for ep in self.episodes.values()
            if action_id in ep.action_distribution
        ]
        # Sort by how often the action was used
        matching.sort(key=lambda ep: ep.action_distribution.get(action_id, 0), reverse=True)
        return matching[:n]
    
    def get_search_episodes(self, n: int = 10) -> List[Episode]:
        """Get episodes with web searches."""
        return [ep for ep in self.episodes.values() if ep.search_count > 0][:n]
    
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get a specific episode by ID."""
        return self.episodes.get(episode_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        if not self.episodes:
            return {
                "total_episodes": 0,
                "total_turns": 0,
            }
        
        rewards = [ep.total_reward for ep in self.episodes.values()]
        avg_rewards = [ep.average_reward for ep in self.episodes.values()]
        turns = [ep.turn_count for ep in self.episodes.values()]
        search_counts = [ep.search_count for ep in self.episodes.values()]
        
        # Action distribution across all episodes
        action_totals = Counter()
        for ep in self.episodes.values():
            action_totals.update(ep.action_distribution)
        
        action_percentages = {}
        total_actions = sum(action_totals.values())
        for action_id, count in action_totals.items():
            name = ACTION_NAMES[action_id] if action_id < len(ACTION_NAMES) else f"ACTION_{action_id}"
            action_percentages[name] = round(count / max(1, total_actions) * 100, 1)
        
        return {
            "total_episodes": len(self.episodes),
            "total_turns": sum(turns),
            "avg_turns_per_episode": round(float(np.mean(turns)), 2),
            "avg_reward": round(float(np.mean(rewards)), 4),
            "avg_episode_reward": round(float(np.mean(avg_rewards)), 4),
            "reward_std": round(float(np.std(rewards)), 4),
            "best_episode_reward": round(float(max(rewards)), 4),
            "worst_episode_reward": round(float(min(rewards)), 4),
            "total_searches": sum(search_counts),
            "episodes_with_search": sum(1 for s in search_counts if s > 0),
            "action_distribution": action_percentages,
            "indexed_keywords": len(self._keyword_index),
        }
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Extract insights that could help improve the agent."""
        insights = {
            "recommendations": [],
            "patterns": [],
        }
        
        # Analyze action success rates
        action_stats = self.get_action_success_rates()
        
        # Find best and worst performing actions
        if action_stats:
            sorted_actions = sorted(
                action_stats.items(),
                key=lambda x: x[1].get("mean_reward", 0),
                reverse=True
            )
            
            best_action = sorted_actions[0]
            worst_action = sorted_actions[-1]
            
            insights["best_action"] = {
                "name": best_action[0],
                "mean_reward": best_action[1]["mean_reward"],
            }
            insights["worst_action"] = {
                "name": worst_action[0],
                "mean_reward": worst_action[1]["mean_reward"],
            }
            
            # Check if search is underutilized
            search_stats = action_stats.get("SEARCH_WEB", {})
            if search_stats.get("count", 0) < 10:
                insights["recommendations"].append(
                    "Web search is underutilized. Consider increasing search exploration."
                )
            elif search_stats.get("mean_reward", 0) > 0.6:
                insights["recommendations"].append(
                    "Web search has high rewards. The agent should learn to use it more."
                )
        
        # Check for search effectiveness
        search_eff = self.get_search_effectiveness()
        if search_eff.get("total_searches", 0) > 0:
            if search_eff.get("success_rate", 0) > 0.7:
                insights["patterns"].append("Web searches are generally successful")
            elif search_eff.get("success_rate", 0) < 0.3:
                insights["patterns"].append("Web searches often don't help - may need better query formulation")
        
        return insights
    
    def _save(self):
        """Save memory to disk."""
        if not self.storage_path:
            return
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Save episodes
        episodes_file = self.storage_path / "episodes.json"
        data = {
            "version": "0.2.0",
            "saved_at": datetime.now().isoformat(),
            "episode_index": self.episode_index,
            "episodes": {ep_id: ep.to_dict() for ep_id, ep in self.episodes.items()},
            "action_reward_history": {str(k): v[-1000:] for k, v in self._action_reward_history.items()},
            "search_query_history": self._search_query_history[-500:],
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
            
            # Load action history
            action_history = data.get("action_reward_history", {})
            for action_id_str, rewards in action_history.items():
                self._action_reward_history[int(action_id_str)] = rewards
            
            # Load search history
            self._search_query_history = data.get("search_query_history", [])
            # Convert to list of tuples if needed
            self._search_query_history = [
                tuple(item) if isinstance(item, list) else item
                for item in self._search_query_history
            ]
                    
        except Exception as e:
            print(f"Warning: Failed to load episodic memory: {e}")
    
    def force_save(self):
        """Force immediate save to disk."""
        if self.storage_path:
            self._save()
            self._episodes_since_save = 0
    
    def export_for_training(self) -> List[Dict[str, Any]]:
        """Export episodes in format suitable for offline RL training."""
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
                    "search_performed": entry.search_performed,
                    "tool_used": entry.tool_used,
                })
        
        return training_data
    
    def export_search_patterns(self) -> List[Dict[str, Any]]:
        """Export patterns of successful search queries."""
        patterns = []
        
        for episode in self.episodes.values():
            for entry in episode.entries:
                if entry.search_performed and entry.search_query:
                    patterns.append({
                        "user_message": entry.user_message,
                        "search_query": entry.search_query,
                        "result_count": entry.search_result_count,
                        "reward": entry.reward,
                        "successful": entry.reward > 0.5,
                    })
        
        return patterns


# ===============================
# Quick Test
# ===============================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Episodic Memory")
    print("=" * 60)
    
    import tempfile
    
    # Create memory with temp storage
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = EpisodicMemory(storage_path=tmpdir, auto_save=True, auto_save_interval=1)
        
        print("\n📝 Creating test episodes...")
        
        # Create multiple test episodes
        for ep_num in range(3):
            memory.start_episode(metadata={"test_episode": ep_num})
            
            for i in range(5):
                # Simulate different actions
                action_id = (ep_num + i) % 7
                action_name = ACTION_NAMES[action_id]
                
                # Search actions get extra data
                search_performed = (action_id == 3)
                search_query = f"test query {i}" if search_performed else None
                
                memory.add_turn(
                    user_message=f"Test message {i} for episode {ep_num}",
                    assistant_response=f"Test response {i}",
                    action_taken=action_id,
                    action_name=action_name,
                    reward=0.3 + (i * 0.1) + (0.2 if search_performed else 0),
                    state_vector=np.random.randn(16),
                    search_performed=search_performed,
                    search_query=search_query,
                    search_result_count=5 if search_performed else 0,
                )
            
            memory.end_episode(summary=f"Test episode {ep_num}")
        
        print(f"\n📊 Memory Stats:")
        stats = memory.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print(f"\n🎯 Action Success Rates:")
        action_stats = memory.get_action_success_rates()
        for action, data in action_stats.items():
            print(f"   {action}: mean={data['mean_reward']:.3f}, count={data['count']}")
        
        print(f"\n🔍 Search Effectiveness:")
        search_eff = memory.get_search_effectiveness()
        print(f"   Total searches: {search_eff['total_searches']}")
        if search_eff['total_searches'] > 0:
            print(f"   Mean reward: {search_eff['mean_reward']:.3f}")
        
        print(f"\n💡 Learning Insights:")
        insights = memory.get_learning_insights()
        if insights.get("best_action"):
            print(f"   Best action: {insights['best_action']['name']}")
        for rec in insights.get("recommendations", []):
            print(f"   → {rec}")
        
        print(f"\n🔎 Search Test:")
        results = memory.search("test message")
        print(f"   Found {len(results)} episodes matching 'test message'")
        
        print(f"\n💾 Export Test:")
        training_data = memory.export_for_training()
        print(f"   Exported {len(training_data)} training samples")
        
        search_patterns = memory.export_search_patterns()
        print(f"   Exported {len(search_patterns)} search patterns")
        
        # Test reload
        print(f"\n🔄 Testing save/reload...")
        memory.force_save()
        
        memory2 = EpisodicMemory(storage_path=tmpdir)
        print(f"   Reloaded {len(memory2.episodes)} episodes")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
