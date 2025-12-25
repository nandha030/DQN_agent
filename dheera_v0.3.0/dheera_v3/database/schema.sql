-- ============================================
-- Dheera v0.3.0 - SQLite Database Schema
-- ============================================

-- 1. EXPERIENCES (Rainbow DQN Replay Buffer)
CREATE TABLE IF NOT EXISTS experiences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state BLOB NOT NULL,
    action INTEGER NOT NULL,
    reward REAL NOT NULL,
    next_state BLOB NOT NULL,
    done INTEGER DEFAULT 0,
    
    -- N-step returns
    n_step_reward REAL,
    n_step_next_state BLOB,
    n_step INTEGER DEFAULT 1,
    
    -- Prioritized replay
    priority REAL DEFAULT 1.0,
    
    -- Curiosity
    intrinsic_reward REAL DEFAULT 0.0,
    novelty_score REAL DEFAULT 0.0,
    
    episode_id TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_exp_priority ON experiences(priority DESC);
CREATE INDEX IF NOT EXISTS idx_exp_episode ON experiences(episode_id);

-- 2. EPISODES (Conversation Sessions)
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    total_reward REAL DEFAULT 0,
    total_intrinsic_reward REAL DEFAULT 0,
    turn_count INTEGER DEFAULT 0,
    summary TEXT,
    
    action_distribution TEXT,  -- JSON
    search_count INTEGER DEFAULT 0,
    tool_count INTEGER DEFAULT 0,
    
    user_id TEXT DEFAULT 'default',
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_ep_user ON episodes(user_id);
CREATE INDEX IF NOT EXISTS idx_ep_time ON episodes(start_time DESC);

-- 3. TURNS (Individual Messages)
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    
    user_message TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    
    -- Action
    action_id INTEGER NOT NULL,
    action_name TEXT,
    action_confidence REAL,
    
    -- State
    state_vector BLOB,
    
    -- Cognitive
    intent TEXT,
    intent_confidence REAL,
    entities TEXT,  -- JSON
    dialogue_state TEXT,  -- JSON
    
    -- Rewards
    immediate_reward REAL DEFAULT 0,
    intrinsic_reward REAL DEFAULT 0,
    human_feedback REAL,
    rlhf_reward REAL,
    
    -- RAG
    rag_context TEXT,
    rag_sources TEXT,  -- JSON
    
    -- Search
    search_performed INTEGER DEFAULT 0,
    search_query TEXT,
    search_results TEXT,  -- JSON
    
    -- Tool
    tool_used TEXT,
    tool_input TEXT,
    tool_output TEXT,
    
    -- Performance
    latency_ms REAL,
    tokens_used INTEGER,
    
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (episode_id) REFERENCES episodes(id)
);

CREATE INDEX IF NOT EXISTS idx_turn_episode ON turns(episode_id, turn_number);

-- 4. PREFERENCES (RLHF Preference Pairs)
CREATE TABLE IF NOT EXISTS preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    turn_id_chosen INTEGER NOT NULL,
    turn_id_rejected INTEGER NOT NULL,
    
    user_message TEXT NOT NULL,
    response_chosen TEXT NOT NULL,
    response_rejected TEXT NOT NULL,
    
    state_vector BLOB,
    preference_strength REAL DEFAULT 1.0,
    
    source TEXT DEFAULT 'human',
    used_in_training INTEGER DEFAULT 0,
    
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (turn_id_chosen) REFERENCES turns(id),
    FOREIGN KEY (turn_id_rejected) REFERENCES turns(id)
);

-- 5. REWARD_MODEL_DATA (RLHF Training)
CREATE TABLE IF NOT EXISTS reward_model_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    state BLOB NOT NULL,
    action INTEGER NOT NULL,
    response_embedding BLOB,
    
    human_reward REAL,
    preference_score REAL,
    
    used_in_training INTEGER DEFAULT 0,
    training_epoch INTEGER,
    
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 6. EMBEDDINGS (Vector Backup)
CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    content_type TEXT NOT NULL,  -- 'turn', 'episode', 'document', 'search'
    content_id TEXT NOT NULL,
    
    embedding BLOB NOT NULL,
    text_content TEXT,
    metadata TEXT,  -- JSON
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_emb_type ON embeddings(content_type, content_id);

-- 7. SEARCH_CACHE
CREATE TABLE IF NOT EXISTS search_cache (
    query_hash TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    results TEXT NOT NULL,  -- JSON
    result_count INTEGER,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME
);

CREATE INDEX IF NOT EXISTS idx_search_expires ON search_cache(expires_at);

-- 8. MODEL_CHECKPOINTS
CREATE TABLE IF NOT EXISTS model_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT NOT NULL,  -- 'rainbow_dqn', 'reward_model', 'rnd'
    checkpoint_path TEXT NOT NULL,
    
    total_steps INTEGER,
    epsilon REAL,
    avg_reward REAL,
    avg_q_value REAL,
    
    config TEXT,  -- JSON
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ckpt_model ON model_checkpoints(model_type, created_at DESC);

-- 9. CURIOSITY_STATES (RND Tracking)
CREATE TABLE IF NOT EXISTS curiosity_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state_hash TEXT NOT NULL,
    state BLOB NOT NULL,
    
    visit_count INTEGER DEFAULT 1,
    avg_prediction_error REAL,
    last_visited DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(state_hash)
);

CREATE INDEX IF NOT EXISTS idx_curiosity_hash ON curiosity_states(state_hash);

-- 10. COGNITIVE_CACHE (Intent/Entity Cache)
CREATE TABLE IF NOT EXISTS cognitive_cache (
    message_hash TEXT PRIMARY KEY,
    message TEXT NOT NULL,
    
    intent TEXT,
    intent_confidence REAL,
    entities TEXT,  -- JSON
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 11. ANALYTICS
CREATE TABLE IF NOT EXISTS analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    
    action_id INTEGER,
    episode_id TEXT,
    
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    time_bucket TEXT
);

CREATE INDEX IF NOT EXISTS idx_analytics ON analytics(metric_name, timestamp DESC);

-- 12. USER_PROFILES
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    
    preferred_actions TEXT,  -- JSON
    topics_of_interest TEXT,  -- JSON
    communication_style TEXT,
    
    total_interactions INTEGER DEFAULT 0,
    positive_feedback_count INTEGER DEFAULT 0,
    negative_feedback_count INTEGER DEFAULT 0,
    
    first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- VIEWS
-- ============================================

-- High priority experiences for sampling
CREATE VIEW IF NOT EXISTS v_priority_experiences AS
SELECT * FROM experiences 
WHERE priority > 0.1 
ORDER BY priority DESC 
LIMIT 10000;

-- Action performance
CREATE VIEW IF NOT EXISTS v_action_stats AS
SELECT 
    action_id,
    action_name,
    COUNT(*) as total_uses,
    AVG(immediate_reward) as avg_reward,
    AVG(human_feedback) as avg_human_feedback,
    AVG(intrinsic_reward) as avg_curiosity,
    AVG(latency_ms) as avg_latency
FROM turns
GROUP BY action_id, action_name;

-- Episode summary
CREATE VIEW IF NOT EXISTS v_episode_summary AS
SELECT 
    e.id,
    e.start_time,
    e.total_reward,
    e.total_intrinsic_reward,
    e.turn_count,
    COUNT(CASE WHEN t.human_feedback > 0 THEN 1 END) as positive_count,
    COUNT(CASE WHEN t.human_feedback < 0 THEN 1 END) as negative_count
FROM episodes e
LEFT JOIN turns t ON e.id = t.episode_id
GROUP BY e.id;

-- Recent curiosity (novel states)
CREATE VIEW IF NOT EXISTS v_novel_states AS
SELECT * FROM curiosity_states
WHERE visit_count = 1
ORDER BY last_visited DESC
LIMIT 100;
