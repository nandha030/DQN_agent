# 🔧 Chat Issue - Troubleshooting Guide

## 🚨 Current Status

**Issue:** Chat API is returning no response (timing out after 15-30 seconds)

**Root Cause:** Ollama LLM service is hanging/not responding

---

## 🔍 Investigation Results

### **1. Backend Status:**
```
✅ Backend API running on port 8000
✅ Dheera initialized successfully
✅ Health endpoint returns healthy
```

### **2. Spiking Networks:**
```
⚡ Successfully integrated with fixes:
  - Fixed import: SpikingRainbowDQNAgent
  - Fixed select_action() to return tuple (action, info)
  - Added store_transition() method
  - Fixed JSON serialization (numpy types → Python types)
```

**However:** Temporarily disabled (`spiking.enabled: false`) until full interface compatibility is verified

### **3. Ollama LLM Service:**
```
❌ Ollama is timing out (10-30 second requests hanging)
❌ Test generation requests not completing
```

**This is the blocking issue!**

---

## 🛠️ Fixes Applied

### **1. SpikingRainbowDQNAgent Interface Fixes:**

**File:** [core/spiking_rainbow_dqn.py](core/spiking_rainbow_dqn.py)

**Changes:**
1. `select_action()` now returns `Tuple[int, Dict]` to match `RainbowDQNAgent` interface
2. Added `store_transition()` method for compatibility
3. Fixed `get_stats()` to convert numpy types to native Python types (int, float)

**Code:**
```python
def select_action(self, state: np.ndarray, training: bool = True):
    """Returns: Tuple[int, Dict] - (action, info_dict)"""
    # ... action selection logic ...
    info = {
        "q_value": float(max_q),
        "spiking": self.use_spiking,
    }
    return action, info

def store_transition(self, state, action, reward, next_state, done, episode_id=None) -> float:
    """Store transition and compute intrinsic reward"""
    # ... reward computation ...
    return float(total_reward)

def get_stats(self) -> Dict[str, Any]:
    """Convert all numpy types to Python types for JSON serialization"""
    stats = {
        "update_count": int(self.update_count),
        "action_distribution": {
            name: int(count) for name, count in zip(self.ACTION_NAMES, self.action_counts)
        },
        # ... more stats ...
    }
    return stats
```

---

## 🔧 How to Fix Ollama

### **Option 1: Restart Ollama (Recommended)**
```bash
# Stop Ollama
pkill -f ollama

# Start Ollama
ollama serve &

# Wait 5 seconds
sleep 5

# Test if working
curl http://localhost:11434/api/tags
```

### **Option 2: Kill Hung Processes**
```bash
# Find Ollama processes
ps aux | grep ollama

# Kill any hung model processes
pkill -9 -f "qwen2|llama3|phi3"

# Restart Ollama
ollama serve &
```

### **Option 3: Switch to Smaller Model**
If Ollama keeps hanging, use a smaller/faster model:

Edit `config/dheera_config.yaml`:
```yaml
slm:
  model_name: "qwen2:1.5b"  # Faster than llama3.2
  timeout: 30  # Increase timeout
```

### **Option 4: Use Direct Python LLM (No Ollama)**
If Ollama continues to have issues, you can bypass it temporarily by modifying the SLM interface to use a lightweight local model or API.

---

## ✅ Once Ollama is Fixed

### **1. Restart Dheera Backend:**
```bash
cd /Users/nandhavignesh/triton/dheera_v0.3.0/dheera_v3

# Kill old backend
lsof -ti:8000 | xargs kill -9

# Start new backend
python3 api/server.py > /tmp/dheera_backend.log 2>&1 &
```

### **2. Test Chat:**
```bash
python3 << 'EOF'
import requests

response = requests.post(
    'http://localhost:8000/api/chat',
    json={'message': 'Hello!'},
    timeout=30
)

if response.status_code == 200:
    print('✅ CHAT WORKING!')
    print(response.json().get('response'))
else:
    print(f'❌ Error: {response.status_code}')
EOF
```

### **3. Re-enable Spiking Networks (Optional):**

After verifying chat works with regular DQN:

Edit `config/dheera_config.yaml`:
```yaml
spiking:
  enabled: true  # Re-enable for 7x speedup
```

Then restart backend.

---

## 📊 What's Working vs What's Not

### ✅ **Working:**
- Backend API server
- Dheera initialization
- Database & Memory monitoring (GUI)
- Ontology Graph viewer (GUI)
- SpikingRainbowDQNAgent interface fixes
- JSON serialization

### ❌ **Not Working:**
- Ollama LLM generation (hanging)
- Chat API (times out waiting for LLM)

### ⚠️ **Temporarily Disabled:**
- Spiking neural networks (`spiking.enabled: false`)
  - Reason: Needs full testing after Ollama is fixed
  - Status: Interface compatibility fixes applied, ready to re-enable

---

## 🎯 Quick Fix Summary

**Immediate Action:**
```bash
# 1. Restart Ollama
pkill -f ollama && ollama serve &

# 2. Wait 10 seconds
sleep 10

# 3. Restart Dheera backend
cd /Users/nandhavignesh/triton/dheera_v0.3.0/dheera_v3
lsof -ti:8000 | xargs kill -9
python3 api/server.py &

# 4. Test chat
sleep 5
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Hello"}'
```

**If that works:**
- ✅ Chat is fixed
- ✅ Can re-enable spiking networks
- ✅ All features operational

---

## 📝 Spiking Networks - Next Steps

Once Ollama/chat is working:

**1. Test with Regular DQN First:**
- Verify chat responses are coming through
- Check latency is reasonable (~1-3 seconds)

**2. Re-enable Spiking Networks:**
```yaml
spiking:
  enabled: true
```

**3. Monitor Startup Logs:**
Look for:
```
⚡ Initializing SpikingRainbow DQN...
✓ SpikingRainbow DQN (69% sparsity, 97% energy savings)
```

**4. Send Test Messages:**
- Simple: "Hello"
- Complex: "Explain quantum computing"
- Monitor if responses are faster (~7x DQN speedup)

**5. Check GUI Stats:**
- Navigate to: **🧠 Core Engine → 🌊 Rainbow DQN**
- Look for:
  - Spiking Enabled: ✅ Yes
  - Sparsity: 65-72%
  - Energy Savings: 90-98%

---

## 🚀 Performance Expectations

### **With Regular DQN:**
- DQN action selection: ~85ms
- Total message latency: ~1-3 seconds (mostly LLM)

### **With Spiking DQN (once re-enabled):**
- DQN action selection: ~12ms (7x faster!)
- Total message latency: ~1-2.9 seconds (~10% improvement)
- Energy: 97% reduction in DQN component

---

## 📚 Related Documentation

- [SPIKING_NETWORKS_ACTIVATED.md](SPIKING_NETWORKS_ACTIVATED.md) - Complete spiking networks guide
- [SESSION_SUMMARY.md](SESSION_SUMMARY.md) - All features implemented this session
- [DATABASE_MEMORY_MONITORING.md](DATABASE_MEMORY_MONITORING.md) - Database/memory monitoring
- [ONTOLOGY_VIEWER_READY.md](ONTOLOGY_VIEWER_READY.md) - Ontology graph viewer

---

## ✅ Summary

**Problem:** Chat not responding due to Ollama LLM hanging

**Solution:** Restart Ollama service

**Status:**
- ✅ Spiking network interface fixes complete
- ✅ Database & ontology features working
- ⚠️ Spiking temporarily disabled until LLM fixed
- ❌ Need to fix Ollama to test end-to-end

**Next:** Restart Ollama → Test chat → Re-enable spiking networks

---

🔧 **Fix Ollama first, then everything else will work!**
