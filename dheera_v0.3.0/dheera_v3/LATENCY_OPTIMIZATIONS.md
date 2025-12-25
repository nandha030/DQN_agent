# ⚡ Dheera Latency Optimization Guide

**Goal**: Reduce response latency from 24-60 seconds to under 5 seconds

## 🎯 Applied Optimizations (DONE)

### 1. **SLM Timeout Reduction** ✅
**Before**: 60 seconds | **After**: 15 seconds
- **File**: `config/dheera_config.yaml:12`
- **Impact**: 4x faster timeout detection
- **Trade-off**: May timeout on complex queries (but phi3:mini is fast)

```yaml
timeout: 15  # Reduced from 60
```

### 2. **Token Limit Reduction** ✅
**Before**: 512 tokens | **After**: 256 tokens
- **File**: `config/dheera_config.yaml:14`
- **Impact**: ~2x faster generation (fewer tokens = less inference time)
- **Trade-off**: Shorter responses (good for chat, not essays)

```yaml
max_tokens: 256  # Reduced from 512
```

### 3. **DQN Training Frequency** ✅
**Before**: Train every 4 steps | **After**: Train every 10 steps
- **File**: `config/dheera_config.yaml:35`
- **Impact**: 2.5x fewer training calls per conversation
- **Trade-off**: Slightly slower learning (acceptable for chat)

```yaml
train_every: 10  # Increased from 4
```

### 4. **Smaller Batch Size** ✅
**Before**: 64 | **After**: 32
- **File**: `config/dheera_config.yaml:27`
- **Impact**: 2x faster training step when it does run
- **Trade-off**: Less stable gradients (but acceptable for RL)

```yaml
batch_size: 32  # Reduced from 64
```

### 5. **Reduced RAG Results** ✅
**Before**: 5 results | **After**: 3 results
- **File**: `config/dheera_config.yaml:74`
- **Impact**: 40% fewer embeddings to compute and rank
- **Trade-off**: Less context (but 3 is usually enough)

```yaml
default_n_results: 3  # Reduced from 5
```

### 6. **Skip RAG for Simple Intents** ✅
**New logic in**: `dheera.py:342-356`
- **Impact**: No RAG retrieval for greetings/thanks/affirmations
- **Savings**: ~500-1000ms per simple query
- **Trade-off**: None (these don't need historical context)

```python
# Skips RAG for: "Hello", "Thanks", "Okay", etc.
skip_rag_intents = {"greeting", "affirmation", "thanks", "farewell"}
```

### 7. **Stricter RAG Filtering** ✅
**Before**: min_score=0.3 | **After**: min_score=0.5
- **File**: `config/dheera_config.yaml:75`
- **Impact**: Fewer low-quality results = faster filtering
- **Trade-off**: Miss some borderline-relevant docs (acceptable)

```yaml
min_score: 0.5  # Increased from 0.3
```

## 📊 Expected Performance Impact

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| SLM Timeout | 60s | 15s | **4x** |
| SLM Generation | ~10s | ~5s | **2x** |
| DQN Training | Every 4 | Every 10 | **2.5x less** |
| RAG Retrieval | 5 docs | 3 docs | **1.67x** |
| Simple Queries | RAG + SLM | SLM only | **2-3x** |

**Total Expected Latency**:
- **Complex queries**: 5-10 seconds (down from 24-60s)
- **Simple greetings**: 2-5 seconds (down from 24-60s)
- **Factual questions**: 7-12 seconds (down from 40-60s)

## 🚀 Additional Optimizations (Optional)

### Level 2: Disable Unused Features

**If you're not using them, disable to save time:**

#### Option A: Disable Goal Evaluator
```yaml
# config/dheera_config.yaml
goal_evaluator:
  enabled: false  # Skip goal evaluation (saves ~100-200ms)
```

#### Option B: Disable Planner
```yaml
# config/dheera_config.yaml
planner:
  enabled: false  # Skip multi-step planning (saves ~200-500ms)
```

#### Option C: Disable Auto-Critic
```yaml
# config/dheera_config.yaml
auto_critic:
  enabled: false  # Skip self-evaluation (saves ~300-500ms)
```

#### Option D: Disable RSI (Self-Improvement)
```yaml
# config/dheera_config.yaml
rsi:
  enabled: false  # Skip RSI checks (saves ~100-200ms)
```

**Total savings if all disabled**: 700-1400ms per turn

### Level 3: Hardware Optimizations

#### Use GPU for Embeddings (if available)
```yaml
# config/dheera_config.yaml
embedding:
  use_gpu: true  # 5-10x faster embeddings (if you have CUDA)
```

#### Use Smaller Embedding Model
```yaml
# config/dheera_config.yaml
embedding:
  model: "all-MiniLM-L6-v2"  # Already optimal (384 dims)
  # Don't use: "all-mpnet-base-v2" (768 dims = slower)
```

#### Use Faster SLM Model
```yaml
# config/dheera_config.yaml
slm:
  model: "phi3:mini"     # Current (good balance)
  # Alternatives:
  # model: "gemma:2b"    # Faster, less capable
  # model: "tinyllama"   # Fastest, lowest quality
  # model: "qwen2:1.5b"  # Fast, decent quality
```

### Level 4: Code-Level Optimizations

#### 4.1: Parallel RAG Retrieval
Currently RAG searches 3 collections **sequentially**. Make it parallel:

```python
# rag/retriever.py:216-232
import concurrent.futures

all_documents = []
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = []
    if "conversations" in sources:
        futures.append(executor.submit(
            self.conversations.search_by_embedding,
            query_embedding, n_results
        ))
    # ... submit all in parallel

    for future in concurrent.futures.as_completed(futures):
        all_documents.extend(future.result())
```

**Savings**: 50-70% of RAG retrieval time

#### 4.2: Cache Intent Classification
```python
# cognitive/intent_classifier.py
from functools import lru_cache

@lru_cache(maxsize=100)
def classify_cached(self, user_message: str):
    # Cache results for identical messages
    return self.classify(user_message)
```

**Savings**: 50-100ms for repeated patterns

#### 4.3: Lazy DB Writes
Write to DB **after** responding to user (non-blocking):

```python
# dheera.py:509 - Use threading for DB writes
import threading

def _async_db_write():
    self.db.store_turn(...)

threading.Thread(target=_async_db_write, daemon=True).start()
```

**Savings**: 200-500ms per turn

## 🎬 Quick Test

Run chat with optimizations:

```bash
python3 run_chat.py --debug
```

Test queries:
1. **Simple**: "Hello!" (should be <5s)
2. **Medium**: "What is Python?" (should be <10s)
3. **Complex**: "Explain quantum computing" (should be <15s)

## 📈 Monitoring Performance

Check stats after 10 turns:

```python
# In chat:
/stats
```

Look for:
- `avg_latency_ms` - Should be under 5000ms
- `slm.avg_latency_ms` - Should be 2000-5000ms
- `rag.avg_retrieval_ms` - Should be under 100ms

## 🔧 Rollback Plan

If responses become too short or quality degrades:

1. **Increase max_tokens back to 512** (line 14)
2. **Reduce train_every back to 4** (line 35)
3. **Keep other optimizations** (they're safe)

## 🎯 Best Configuration for Chat

**For interactive chat** (prioritize speed):
```yaml
slm:
  timeout: 10          # Even faster!
  max_tokens: 128      # Short chat responses

dqn:
  train_every: 20      # Train even less frequently

rag:
  default_n_results: 2 # Minimal context
```

**For knowledge Q&A** (prioritize quality):
```yaml
slm:
  timeout: 30
  max_tokens: 512

dqn:
  train_every: 4

rag:
  default_n_results: 5
  min_score: 0.3
```

## 🚨 Troubleshooting

### Issue: Responses still slow (>15s)

**Check 1**: Is Ollama running locally?
```bash
ollama list  # Should show phi3:mini
```

**Check 2**: Is Ollama overloaded?
```bash
# Use smaller model
ollama pull gemma:2b
# Update config to use gemma:2b
```

**Check 3**: Check actual bottleneck
```python
# Add timing to dheera.py
import time

# Before each major step:
t1 = time.time()
# ... code ...
print(f"Step X took: {(time.time()-t1)*1000:.0f}ms")
```

### Issue: Responses too short

**Solution**: Increase max_tokens incrementally
```yaml
max_tokens: 256  # Start here
max_tokens: 384  # If too short
max_tokens: 512  # Max reasonable
```

### Issue: Quality degraded

**Solution**: Re-enable critical components
```yaml
goal_evaluator:
  enabled: true  # Helps with complex queries

auto_critic:
  enabled: true  # Helps catch errors
```

## 📚 Summary

**✅ Applied (Active Now)**:
- 15s timeout (was 60s)
- 256 tokens (was 512)
- Train every 10 steps (was 4)
- 3 RAG results (was 5)
- Skip RAG for greetings
- Stricter RAG filtering (0.5 vs 0.3)

**🔧 Optional (Your Choice)**:
- Disable unused modules (goal/planner/critic)
- Use GPU for embeddings
- Switch to faster model (gemma:2b)
- Implement parallel RAG
- Cache intent classification
- Async DB writes

**🎯 Expected Result**: **5-10 second** responses (down from 24-60s)

---

**Next Steps**:
1. Test with `python3 run_chat.py --debug`
2. Monitor latency with `/stats`
3. Adjust `max_tokens` if responses too short
4. Consider Level 2-4 optimizations if still slow
