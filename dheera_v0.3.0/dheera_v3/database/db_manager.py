# database/db_manager.py
"""
Dheera v0.3.0 - SQLite Database Manager
Handles all database operations for experiences, episodes, RLHF, and more.
"""

import sqlite3
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager
import uuid


class DheeraDatabase:
    """
    SQLite database manager for Dheera v0.3.0
    
    Handles:
    - Experience replay buffer (Rainbow DQN)
    - Episode and turn storage
    - RLHF preferences
    - Embeddings backup
    - Curiosity state tracking
    - Analytics
    """
    
    def __init__(self, db_path: str = "dheera.db"):
        self.db_path = db_path
        self.schema_path = Path(__file__).parent / "schema.sql"
        self._init_db()
    
    def _init_db(self):
        """Initialize database with schema."""
        with self._get_conn() as conn:
            if self.schema_path.exists():
                with open(self.schema_path, 'r') as f:
                    conn.executescript(f.read())
            conn.commit()
    
    @contextmanager
    def _get_conn(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # ==================== EXPERIENCES (Rainbow DQN) ====================
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float = 1.0,
        intrinsic_reward: float = 0.0,
        novelty_score: float = 0.0,
        episode_id: Optional[str] = None,
        n_step_reward: Optional[float] = None,
        n_step_next_state: Optional[np.ndarray] = None,
        n_step: int = 1,
    ) -> int:
        """Store a single experience for replay."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiences 
                (state, action, reward, next_state, done, priority, 
                 intrinsic_reward, novelty_score, episode_id,
                 n_step_reward, n_step_next_state, n_step)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                state.tobytes(),
                action,
                reward,
                next_state.tobytes(),
                int(done),
                priority,
                intrinsic_reward,
                novelty_score,
                episode_id,
                n_step_reward,
                n_step_next_state.tobytes() if n_step_next_state is not None else None,
                n_step,
            ))
            conn.commit()
            return cursor.lastrowid
    
    def sample_experiences(
        self,
        batch_size: int,
        prioritized: bool = True,
        state_dim: int = 64,
    ) -> Tuple[List[Dict], List[int], np.ndarray]:
        """
        Sample batch of experiences for training.
        
        Returns:
            (experiences, indices, weights)
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            if prioritized:
                # Prioritized sampling
                cursor.execute("""
                    SELECT id, state, action, reward, next_state, done, priority,
                           intrinsic_reward, n_step_reward, n_step_next_state, n_step
                    FROM experiences
                    ORDER BY priority DESC
                    LIMIT ?
                """, (batch_size * 3,))  # Sample more, then random select
                
                rows = cursor.fetchall()
                if len(rows) < batch_size:
                    return [], [], np.array([])
                
                # Compute sampling probabilities
                priorities = np.array([row['priority'] for row in rows])
                probs = priorities / priorities.sum()
                
                # Sample indices
                indices = np.random.choice(len(rows), size=batch_size, p=probs, replace=False)
                sampled_rows = [rows[i] for i in indices]
                
                # Compute importance sampling weights
                weights = (len(rows) * probs[indices]) ** (-0.4)  # beta = 0.4
                weights = weights / weights.max()
            else:
                # Uniform sampling
                cursor.execute("""
                    SELECT id, state, action, reward, next_state, done, priority,
                           intrinsic_reward, n_step_reward, n_step_next_state, n_step
                    FROM experiences
                    ORDER BY RANDOM()
                    LIMIT ?
                """, (batch_size,))
                
                sampled_rows = cursor.fetchall()
                weights = np.ones(len(sampled_rows))
            
            # Parse experiences
            experiences = []
            exp_indices = []
            
            for row in sampled_rows:
                exp = {
                    'state': np.frombuffer(row['state'], dtype=np.float32).reshape(state_dim),
                    'action': row['action'],
                    'reward': row['reward'],
                    'next_state': np.frombuffer(row['next_state'], dtype=np.float32).reshape(state_dim),
                    'done': bool(row['done']),
                    'intrinsic_reward': row['intrinsic_reward'],
                    'n_step_reward': row['n_step_reward'],
                    'n_step': row['n_step'],
                }
                
                if row['n_step_next_state']:
                    exp['n_step_next_state'] = np.frombuffer(
                        row['n_step_next_state'], dtype=np.float32
                    ).reshape(state_dim)
                
                experiences.append(exp)
                exp_indices.append(row['id'])
            
            return experiences, exp_indices, weights
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for prioritized replay."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            for idx, priority in zip(indices, priorities):
                cursor.execute(
                    "UPDATE experiences SET priority = ? WHERE id = ?",
                    (priority, idx)
                )
            conn.commit()
    
    def get_experience_count(self) -> int:
        """Get total number of experiences."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM experiences")
            return cursor.fetchone()[0]
    
    def cleanup_old_experiences(self, max_count: int = 100000):
        """Remove old experiences to maintain buffer size."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM experiences 
                WHERE id NOT IN (
                    SELECT id FROM experiences 
                    ORDER BY priority DESC, timestamp DESC 
                    LIMIT ?
                )
            """, (max_count,))
            conn.commit()
            return cursor.rowcount
    
    # ==================== EPISODES ====================
    
    def create_episode(
        self,
        episode_id: Optional[str] = None,
        user_id: str = "default",
        metadata: Optional[Dict] = None,
    ) -> str:
        """Create a new conversation episode."""
        episode_id = episode_id or str(uuid.uuid4())
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO episodes (id, start_time, user_id, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                episode_id,
                datetime.now().isoformat(),
                user_id,
                json.dumps(metadata) if metadata else None,
            ))
            conn.commit()
        
        return episode_id
    
    def end_episode(
        self,
        episode_id: str,
        summary: str = "",
        action_distribution: Optional[Dict[int, int]] = None,
    ):
        """Mark episode as complete."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # Get aggregated stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as turn_count,
                    SUM(immediate_reward) as total_reward,
                    SUM(intrinsic_reward) as total_intrinsic,
                    SUM(search_performed) as search_count,
                    COUNT(tool_used) as tool_count
                FROM turns WHERE episode_id = ?
            """, (episode_id,))
            
            stats = cursor.fetchone()
            
            cursor.execute("""
                UPDATE episodes SET
                    end_time = ?,
                    summary = ?,
                    turn_count = ?,
                    total_reward = ?,
                    total_intrinsic_reward = ?,
                    search_count = ?,
                    tool_count = ?,
                    action_distribution = ?
                WHERE id = ?
            """, (
                datetime.now().isoformat(),
                summary,
                stats['turn_count'],
                stats['total_reward'] or 0,
                stats['total_intrinsic'] or 0,
                stats['search_count'] or 0,
                stats['tool_count'] or 0,
                json.dumps(action_distribution) if action_distribution else None,
                episode_id,
            ))
            conn.commit()
    
    def get_episode(self, episode_id: str) -> Optional[Dict]:
        """Get episode by ID."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # ==================== TURNS ====================
    
    def store_turn(
        self,
        episode_id: str,
        user_message: str,
        assistant_response: str,
        action_id: int,
        action_name: str,
        state_vector: Optional[np.ndarray] = None,
        intent: Optional[str] = None,
        intent_confidence: Optional[float] = None,
        entities: Optional[Dict] = None,
        dialogue_state: Optional[Dict] = None,
        immediate_reward: float = 0.0,
        intrinsic_reward: float = 0.0,
        rag_context: Optional[str] = None,
        rag_sources: Optional[List] = None,
        search_performed: bool = False,
        search_query: Optional[str] = None,
        search_results: Optional[Dict] = None,
        tool_used: Optional[str] = None,
        tool_input: Optional[str] = None,
        tool_output: Optional[str] = None,
        latency_ms: Optional[float] = None,
        tokens_used: Optional[int] = None,
    ) -> int:
        """Store a conversation turn."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # Get next turn number
            cursor.execute(
                "SELECT COALESCE(MAX(turn_number), 0) + 1 FROM turns WHERE episode_id = ?",
                (episode_id,)
            )
            turn_number = cursor.fetchone()[0]
            
            cursor.execute("""
                INSERT INTO turns (
                    episode_id, turn_number, user_message, assistant_response,
                    action_id, action_name, state_vector,
                    intent, intent_confidence, entities, dialogue_state,
                    immediate_reward, intrinsic_reward,
                    rag_context, rag_sources,
                    search_performed, search_query, search_results,
                    tool_used, tool_input, tool_output,
                    latency_ms, tokens_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode_id,
                turn_number,
                user_message,
                assistant_response,
                action_id,
                action_name,
                state_vector.tobytes() if state_vector is not None else None,
                intent,
                intent_confidence,
                json.dumps(entities) if entities else None,
                json.dumps(dialogue_state) if dialogue_state else None,
                immediate_reward,
                intrinsic_reward,
                rag_context,
                json.dumps(rag_sources) if rag_sources else None,
                int(search_performed),
                search_query,
                json.dumps(search_results) if search_results else None,
                tool_used,
                tool_input,
                tool_output,
                latency_ms,
                tokens_used,
            ))
            conn.commit()
            return cursor.lastrowid
    
    def update_turn_feedback(self, turn_id: int, human_feedback: float):
        """Update turn with human feedback."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE turns SET human_feedback = ? WHERE id = ?",
                (human_feedback, turn_id)
            )
            conn.commit()
    
    def get_recent_turns(self, episode_id: str, limit: int = 10) -> List[Dict]:
        """Get recent turns for an episode."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM turns 
                WHERE episode_id = ?
                ORDER BY turn_number DESC
                LIMIT ?
            """, (episode_id, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== RLHF PREFERENCES ====================
    
    def store_preference(
        self,
        turn_id_chosen: int,
        turn_id_rejected: int,
        user_message: str,
        response_chosen: str,
        response_rejected: str,
        state_vector: Optional[np.ndarray] = None,
        preference_strength: float = 1.0,
        source: str = "human",
    ) -> int:
        """Store a preference pair for RLHF training."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO preferences (
                    turn_id_chosen, turn_id_rejected,
                    user_message, response_chosen, response_rejected,
                    state_vector, preference_strength, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                turn_id_chosen,
                turn_id_rejected,
                user_message,
                response_chosen,
                response_rejected,
                state_vector.tobytes() if state_vector is not None else None,
                preference_strength,
                source,
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_preference_batch(
        self,
        batch_size: int,
        unused_only: bool = True,
    ) -> List[Dict]:
        """Get batch of preferences for RLHF training."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM preferences"
            if unused_only:
                query += " WHERE used_in_training = 0"
            query += f" ORDER BY RANDOM() LIMIT {batch_size}"
            
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]
    
    def mark_preferences_used(self, preference_ids: List[int], epoch: int):
        """Mark preferences as used in training."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "UPDATE preferences SET used_in_training = 1 WHERE id = ?",
                [(pid,) for pid in preference_ids]
            )
            conn.commit()
    
    def get_preference_count(self) -> int:
        """Get total preference count."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM preferences")
            return cursor.fetchone()[0]
    
    # ==================== CURIOSITY TRACKING ====================
    
    def update_curiosity_state(
        self,
        state: np.ndarray,
        prediction_error: float,
    ) -> Tuple[int, bool]:
        """
        Update curiosity state tracking.
        
        Returns:
            (visit_count, is_novel)
        """
        state_hash = hashlib.md5(state.tobytes()).hexdigest()
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # Check if state exists
            cursor.execute(
                "SELECT visit_count, avg_prediction_error FROM curiosity_states WHERE state_hash = ?",
                (state_hash,)
            )
            row = cursor.fetchone()
            
            if row:
                # Update existing
                new_count = row['visit_count'] + 1
                new_avg = (row['avg_prediction_error'] * row['visit_count'] + prediction_error) / new_count
                
                cursor.execute("""
                    UPDATE curiosity_states 
                    SET visit_count = ?, avg_prediction_error = ?, last_visited = ?
                    WHERE state_hash = ?
                """, (new_count, new_avg, datetime.now().isoformat(), state_hash))
                conn.commit()
                
                return new_count, False
            else:
                # New state
                cursor.execute("""
                    INSERT INTO curiosity_states (state_hash, state, visit_count, avg_prediction_error)
                    VALUES (?, ?, 1, ?)
                """, (state_hash, state.tobytes(), prediction_error))
                conn.commit()
                
                return 1, True
    
    def get_novel_state_count(self) -> int:
        """Get count of states visited only once."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM curiosity_states WHERE visit_count = 1")
            return cursor.fetchone()[0]
    
    # ==================== EMBEDDINGS ====================
    
    def store_embedding(
        self,
        id: str,
        content_type: str,
        content_id: str,
        embedding: np.ndarray,
        text_content: str,
        metadata: Optional[Dict] = None,
    ):
        """Store embedding for backup."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings 
                (id, content_type, content_id, embedding, text_content, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                id,
                content_type,
                content_id,
                embedding.tobytes(),
                text_content,
                json.dumps(metadata) if metadata else None,
            ))
            conn.commit()
    
    def get_embedding(self, id: str, dim: int = 384) -> Optional[np.ndarray]:
        """Retrieve embedding by ID."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM embeddings WHERE id = ?", (id,))
            row = cursor.fetchone()
            if row:
                return np.frombuffer(row['embedding'], dtype=np.float32).reshape(dim)
            return None
    
    # ==================== SEARCH CACHE ====================
    
    def cache_search(
        self,
        query: str,
        results: Dict,
        ttl_seconds: int = 300,
    ):
        """Cache search results."""
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()
        expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO search_cache 
                (query_hash, query, results, result_count, expires_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                query_hash,
                query,
                json.dumps(results),
                results.get('result_count', 0),
                expires_at.isoformat(),
            ))
            conn.commit()
    
    def get_cached_search(self, query: str) -> Optional[Dict]:
        """Get cached search results if not expired."""
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT results FROM search_cache 
                WHERE query_hash = ? AND expires_at > ?
            """, (query_hash, datetime.now().isoformat()))
            
            row = cursor.fetchone()
            if row:
                return json.loads(row['results'])
            return None
    
    def cleanup_expired_cache(self):
        """Remove expired cache entries."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM search_cache WHERE expires_at < ?",
                (datetime.now().isoformat(),)
            )
            conn.commit()
            return cursor.rowcount
    
    # ==================== CHECKPOINTS ====================
    
    def save_checkpoint_metadata(
        self,
        model_type: str,
        checkpoint_path: str,
        total_steps: int,
        epsilon: Optional[float] = None,
        avg_reward: Optional[float] = None,
        avg_q_value: Optional[float] = None,
        config: Optional[Dict] = None,
    ) -> int:
        """Save checkpoint metadata."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_checkpoints 
                (model_type, checkpoint_path, total_steps, epsilon, avg_reward, avg_q_value, config)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                model_type,
                checkpoint_path,
                total_steps,
                epsilon,
                avg_reward,
                avg_q_value,
                json.dumps(config) if config else None,
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_latest_checkpoint(self, model_type: str) -> Optional[Dict]:
        """Get latest checkpoint for a model type."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM model_checkpoints 
                WHERE model_type = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (model_type,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # ==================== ANALYTICS ====================
    
    def record_metric(
        self,
        metric_name: str,
        metric_value: float,
        action_id: Optional[int] = None,
        episode_id: Optional[str] = None,
    ):
        """Record an analytics metric."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analytics (metric_name, metric_value, action_id, episode_id)
                VALUES (?, ?, ?, ?)
            """, (metric_name, metric_value, action_id, episode_id))
            conn.commit()
    
    def get_action_stats(self) -> List[Dict]:
        """Get action performance statistics."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM v_action_stats")
            return [dict(row) for row in cursor.fetchall()]
    
    def get_episode_summary(self, limit: int = 10) -> List[Dict]:
        """Get recent episode summaries."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM v_episode_summary ORDER BY start_time DESC LIMIT {limit}")
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== UTILITIES ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            tables = ['experiences', 'episodes', 'turns', 'preferences', 'embeddings', 'curiosity_states']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            
            # Additional stats
            stats['novel_states'] = self.get_novel_state_count()
            
            cursor.execute("SELECT AVG(priority) FROM experiences")
            stats['avg_priority'] = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT SUM(total_reward) FROM episodes")
            stats['total_reward'] = cursor.fetchone()[0] or 0
            
            return stats
    
    def vacuum(self):
        """Optimize database file size."""
        with self._get_conn() as conn:
            conn.execute("VACUUM")


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing DheeraDatabase...")
    
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        
        # Copy schema
        schema_content = open(Path(__file__).parent / "schema.sql").read()
        
        db = DheeraDatabase(db_path)
        
        # Test experience storage
        state = np.random.randn(64).astype(np.float32)
        next_state = np.random.randn(64).astype(np.float32)
        
        exp_id = db.store_experience(
            state=state,
            action=3,
            reward=0.5,
            next_state=next_state,
            done=False,
            priority=0.8,
            intrinsic_reward=0.1,
        )
        print(f"✓ Stored experience: {exp_id}")
        
        # Test episode
        episode_id = db.create_episode(user_id="test_user")
        print(f"✓ Created episode: {episode_id}")
        
        # Test turn
        turn_id = db.store_turn(
            episode_id=episode_id,
            user_message="Hello!",
            assistant_response="Hi there!",
            action_id=0,
            action_name="DIRECT_RESPONSE",
            intent="greeting",
            immediate_reward=0.3,
        )
        print(f"✓ Stored turn: {turn_id}")
        
        # Test feedback
        db.update_turn_feedback(turn_id, 1.0)
        print("✓ Updated feedback")
        
        # Test curiosity
        visit_count, is_novel = db.update_curiosity_state(state, 0.5)
        print(f"✓ Curiosity update: visits={visit_count}, novel={is_novel}")
        
        # Test search cache
        db.cache_search("test query", {"results": [1, 2, 3], "result_count": 3})
        cached = db.get_cached_search("test query")
        print(f"✓ Search cache: {cached is not None}")
        
        # Test stats
        stats = db.get_stats()
        print(f"✓ Stats: {stats}")
        
        print("\n✅ All database tests passed!")
