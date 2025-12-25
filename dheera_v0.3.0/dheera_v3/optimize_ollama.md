# 🚀 Ollama Optimization Guide

## Current Model Performance

You're using **phi3:mini (2.2GB)** which is excellent for quality but moderate speed.

## Speed Up Current Model (phi3:mini)

### 1. Set Number of GPU Layers (if you get GPU later)
```bash
# Check if Ollama sees your GPU
ollama show phi3:mini

# Use GPU acceleration (when available)
export OLLAMA_NUM_GPU=999  # Use all GPU layers
ollama serve
```

### 2. Reduce Context Window
Create a custom modelfile:

```bash
# Create Modelfile
cat > Modelfile <<EOF
FROM phi3:mini

# Reduce context for faster inference
PARAMETER num_ctx 2048        # Default is 4096 (reduce by 50%)

# Faster sampling
PARAMETER num_predict 256     # Match your config
PARAMETER top_k 20            # Reduce from 40
PARAMETER top_p 0.9           # Standard
PARAMETER temperature 0.7     # Match your config

# Better performance
PARAMETER num_thread 8        # Use 8 CPU threads (adjust to your CPU)
PARAMETER repeat_penalty 1.1
EOF

# Create optimized model
ollama create phi3-fast -f Modelfile

# Test it
ollama run phi3-fast "Hello"
```

**Update Dheera config:**
```yaml
slm:
  model: "phi3-fast"  # Your optimized version
```

**Expected speedup**: 1.3-1.5x faster (4-5s → 3-4s)

### 3. Use Quantized Model (Already Using)

Your `phi3:mini` is already quantized (Q4_K_M). If you want even faster:

```bash
# Smaller quantization (faster but lower quality)
ollama pull phi3:mini-q2  # If available (not recommended)
```

## Switch to Faster Model

### Option A: Gemma 2B (Recommended)
```bash
ollama pull gemma:2b
```

**Pros:**
- 2-3x faster than phi3:mini
- Great quality (Google)
- Still smart enough for chat

**Cons:**
- Slightly less capable on complex reasoning

### Option B: Qwen2 1.5B (Maximum Speed)
```bash
ollama pull qwen2:1.5b
```

**Pros:**
- 3-4x faster than phi3:mini
- Smallest good model
- Excellent for simple chat

**Cons:**
- Less capable on complex queries
- May hallucinate more

### Option C: Llama3.2 3B (Best Balance)
```bash
ollama pull llama3.2:3b
```

**Pros:**
- Similar speed to phi3:mini
- Meta's latest small model
- Great instruction following

**Cons:**
- About same speed (no improvement)

## Benchmark Your Current Setup

Test current performance:

```bash
# Simple speed test
time ollama run phi3:mini "Say hello in one word" --verbose

# More detailed
ollama run phi3:mini "Hello" --verbose 2>&1 | grep -E "total duration|eval duration"
```

Example output:
```
total duration: 3.2s          ← Total time
eval duration: 2.8s           ← Generation time (main bottleneck)
```

## Full Optimization Stack

### Current Latency Breakdown:
```
Total: 5-10s
├─ Cognitive (intent/entity): ~100ms
├─ RAG retrieval: ~50-100ms (now optimized)
├─ DQN action selection: ~50ms
├─ Ollama SLM: 3000-5000ms ← MAIN BOTTLENECK
├─ DQN training: ~100-200ms (every 10 turns)
└─ DB writes: ~100ms
```

### With Gemma 2B:
```
Total: 3-7s
├─ Cognitive: ~100ms
├─ RAG: ~50ms (optimized + skipped for greetings)
├─ DQN: ~50ms
├─ Ollama SLM: 2000-3000ms ← REDUCED
├─ DQN training: ~100ms (every 10 turns)
└─ DB: ~100ms
```

### With GPU (when available) + Gemma 2B:
```
Total: 2-5s
├─ Cognitive: ~50ms (GPU embeddings)
├─ RAG: ~20ms (GPU embeddings + optimized)
├─ DQN: ~10ms (GPU)
├─ Ollama SLM: 1000-2000ms ← GPU accelerated
├─ DQN training: ~20ms (GPU, every 10 turns)
└─ DB: ~100ms
```

## My Recommendation for You

### Short Term (Now):
1. **Keep phi3:mini** for quality
2. **Create phi3-fast** optimized version (see Modelfile above)
3. **Expected**: 5-10s → 4-7s

### Medium Term (Next Week):
1. **Try gemma:2b** for speed
2. **Compare quality** vs phi3:mini
3. **Expected**: 5-10s → 3-7s

### Long Term (If Needed):
1. **Get GPU** (NVIDIA RTX 3060/4060 or better)
2. **Use gemma:2b with GPU**
3. **Expected**: 5-10s → 2-4s

## Quick Setup Script

```bash
#!/bin/bash
# optimize_ollama.sh

echo "🚀 Optimizing Ollama for Dheera"

# Pull fast model
echo "📥 Pulling gemma:2b..."
ollama pull gemma:2b

# Create optimized phi3
echo "⚙️  Creating phi3-fast..."
cat > /tmp/Modelfile <<EOF
FROM phi3:mini
PARAMETER num_ctx 2048
PARAMETER num_predict 256
PARAMETER top_k 20
PARAMETER num_thread 8
PARAMETER temperature 0.7
EOF

ollama create phi3-fast -f /tmp/Modelfile

echo "✅ Done! You now have:"
echo "  - gemma:2b (fastest, good quality)"
echo "  - phi3-fast (optimized current model)"
echo "  - phi3:mini (original, best quality)"

echo ""
echo "Update config/dheera_config.yaml:"
echo "  model: \"gemma:2b\"     # For speed"
echo "  model: \"phi3-fast\"   # For balanced"
echo "  model: \"phi3:mini\"   # For quality"
```

Make executable and run:
```bash
chmod +x optimize_ollama.sh
./optimize_ollama.sh
```

## Test Performance

After changing model, run benchmark:
```bash
python3 benchmark_latency.py
```

Compare results and pick your favorite!
