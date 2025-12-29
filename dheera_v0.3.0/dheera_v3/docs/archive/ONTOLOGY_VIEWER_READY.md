# ✅ Ontology Graph Viewer - Ready to Use!

## 🎉 What Was Added

An interactive **🧠 Ontology Graph** viewer has been added as the 4th tab in the **🗺️ System Map** page!

---

## 🎯 How to Access

### **Step 1: Restart Streamlit**
```bash
pkill -f streamlit
cd /Users/nandhavignesh/triton/dheera_v0.3.0/dheera_v3
streamlit run gui_professional.py --server.port 8501
```

### **Step 2: Navigate**
1. Open http://localhost:8501
2. Sidebar → Click **"🗺️ System Map"**
3. Top tabs → Click **"🧠 Ontology Graph"**

---

## 📊 5 Interactive Views

### **1. 📚 Classes (Concepts)**
**What you'll see:**
- All 18 ontology classes in a hierarchical tree
- 6 top-level classes:
  - 🧠 Cognitive_Component
  - 💬 Interaction_Unit
  - 🎯 Decision_Artifact
  - 📚 Knowledge_Artifact
  - 🔧 System_Resource
  - ⚖️ Constraint
- 12 subclasses with examples
- Statistics: Total classes, top-level, subclasses

**Example:**
```
🧠 Cognitive_Component (abstract)
├── Learning_Agent
│   └── Example: RainbowDQN_Agent, RLHF_Module
├── Knowledge_Store
│   └── Example: RAG_Engine, Vector_Store
├── Reasoning_Engine
│   └── Example: Curiosity_ICM, Goal_Evaluator
└── Perception_Module
    └── Example: Intent_Classifier, Entity_Extractor
```

---

### **2. 🎯 Entities (Instances)**
**What you'll see:**
- Real system components running in Dheera
- 6 categories to explore:
  - Cognitive Components
  - Interaction Units
  - Decision Artifacts
  - Knowledge Artifacts
  - System Resources
  - Constraints

**Example - RainbowDQN_Agent:**
```
Entity: RainbowDQN_Agent
Type: Learning_Agent
Properties:
  - state_dim: 64
  - action_space: 8
  - learning_rate: 0.0001
  - gamma: 0.99
Constraints:
  - Must train every 10 steps
  - Requires Experience_Replay_Buffer
  - Outputs Action + Q_Values
```

---

### **3. 🔗 Relationships**
**What you'll see:**
- 9 semantic verbs that connect entities
- Domain, range, constraints for each
- Logic/rules that govern the relationship
- Real examples

**Relationships:**
- `trains` - Learning_Agent → Policy
- `retrieves_from` - Turn → Knowledge_Store
- `executed_by` - Action → Cognitive_Component
- `generates` - Component → Decision_Artifact
- `depends_on` - Component → System_Resource
- `constrains` - Constraint → Interaction_Unit
- `feeds_to` - Knowledge_Artifact → Component
- `influences` - Reasoning_Engine → Decision_Artifact
- `stores` - System_Resource → Artifacts

**Example - constrains:**
```
constrains
Domain: Constraint
Range: Interaction_Unit | Decision_Artifact

Constraints:
- Checked BEFORE execution
- Violation blocks action

Logic:
IF constraint violated:
  block()
  log()
  safe_response()

Example:
Input_Policy_Guard constrains User_Message
- Scans for PII
- IF detected → blocks
- Returns: "I cannot process personal information"
```

---

### **4. ⚙️ Inference Rules**
**What you'll see:**
- 17 reasoning rules across 5 categories
- IF-THEN-WHY logic
- Categories:
  - 🧠 Learning Rules (DQN, Curiosity, RLHF)
  - 📚 Retrieval Rules (RAG skip, search forcing, relevance)
  - 🔒 Safety Rules (PII, identity, unsafe code)
  - 🔧 Resource Management (buffer cleanup, cache, fallback)
  - 🎯 Action Selection (heuristics, goals, exploration)

**Example - Curiosity_Reward_Boost:**
```
RULE: Curiosity_Reward_Boost
IF:
  - State is novel (visit_count == 1)
  - Prediction_Error > threshold
THEN:
  - intrinsic_reward = beta * prediction_error
  - total_reward = extrinsic + intrinsic
WHY: Encourage exploration of unknown states
```

---

### **5. 🔍 Reasoning Examples**
**What you'll see:**
- 3 complete reasoning traces
- Step-by-step ontology application
- Color-coded execution flow
- Examples:
  1. **Novel Query Handling** ("What is quantum computing?")
  2. **PII Detection & Blocking** ("My email is john@example.com")
  3. **Search vs RAG Decision** ("Who won the 2024 US election?")

**Example - Novel Query (11 steps):**
```
1️⃣ Create Turn → Turn_123 (intent: factual_question)
2️⃣ Apply RAG Rule → retrieve() from RAG_Engine
3️⃣ RAG Retrieval → doc_001 (score: 0.82) ✅
4️⃣ Encode State → State_Vector_ABC (64-dim)
5️⃣ Check Novelty → NOVEL! intrinsic_reward: +0.146
6️⃣ DQN Action → DIRECT_RESPONSE
7️⃣ LLM Generate → "Quantum computing is..."
8️⃣ Safety Check → ✅ No violations
9️⃣ User Feedback → 👍 reward: 1.146 (high!)
🔟 Store Experience → (state, action, reward, next_state)
1️⃣1️⃣ Train DQN → 50 % 10 == 0 → TRAIN()

Outcome: Dheera learned that novel quantum queries
         → DIRECT_RESPONSE works well
```

---

## 🎯 Quick Stats Display

At the bottom of every view:
- **Total Classes:** 18
- **Entities:** 15+
- **Relationships:** 9
- **Inference Rules:** 17
- **Constraints:** 40+
- **Categories:** 5

---

## 🔍 What Makes This Different from Other Diagrams?

### **Regular System Diagram:**
"RainbowDQN connects to Experience Buffer"

### **Ontology Graph (This):**
```
RainbowDQN_Agent (Learning_Agent)
  |
  | trains (every 10 steps, requires buffer)
  |
  V
Current_Policy (Policy)

RULE: DQN_Training_Trigger
IF total_steps % 10 == 0 AND buffer.count >= 50:
  THEN train()

WHY: Periodic training ensures policy improvement
```

**Difference:**
- ❌ Data visualization → Shows what happened
- ✅ Ontology graph → Shows WHY, HOW, WHEN, and UNDER WHAT CONSTRAINTS

---

## 📚 Use Cases

### **1. Understand How Dheera Thinks**
- Navigate to **🔍 Reasoning Example**
- Select "Novel Query Handling"
- See exact 11-step reasoning process
- Understand WHY each decision was made

### **2. Learn System Architecture**
- Go to **📚 Classes**
- See all 18 classes hierarchically
- Understand what types of entities exist
- Examples for each class

### **3. Explore Real Components**
- Go to **🎯 Entities**
- Select "Cognitive Components"
- See RainbowDQN_Agent configuration
- Properties: state_dim, learning_rate, gamma
- Constraints: training frequency, requirements

### **4. Understand Relationships**
- Go to **🔗 Relationships**
- Select "constrains"
- See how Policy_Guard blocks unsafe inputs
- Logic: IF PII detected → block
- Example: Email detection

### **5. Study Reasoning Rules**
- Go to **⚙️ Inference Rules**
- Select "Safety Rules"
- See Input_PII_Detection rule
- Understand privacy protection logic

---

## 🎨 Visual Features

### **Color Coding:**
- 🔵 Blue (#667eea): State/Input processing
- 🟢 Green (#43a047): Successful operations
- 🟠 Orange (#ff6f00): LLM generation
- 🔴 Red (#e91533): Blocking/Safety
- 🟣 Purple (#f093fb): Novelty/Curiosity
- 🔷 Light Blue (#1e88e5): RAG/Knowledge

### **Interactive Elements:**
- **Radio buttons:** Switch between 5 views
- **Dropdowns:** Filter by category/rule type
- **Expandable sections:** Detailed information
- **Code blocks:** Formatted logic/rules
- **Colored boxes:** Step-by-step traces
- **Metrics:** Quick stats

---

## 🚀 Next Steps (Optional)

### **Phase 1: Current (✅ Complete)**
- Visual ontology viewer
- 5 interactive views
- 3 reasoning examples
- Full documentation

### **Phase 2: Dynamic Ontology (Future)**
- Load ontology from JSON/YAML file
- Real-time rule editing
- Add custom classes/entities
- Export to OWL/RDF format

### **Phase 3: Live Reasoning Trace (Future)**
- Capture actual reasoning during chat
- Show which rules fired
- Display decision tree
- Replay reasoning step-by-step

### **Phase 4: Graph Visualization (Future)**
- Neo4j integration
- Interactive network graph
- Node/edge exploration
- Zoom, pan, filter

---

## 📖 Documentation

- **[ONTOLOGY_GRAPH.md](ONTOLOGY_GRAPH.md)** - Complete ontology specification
- **[SYSTEM_PROMPT_INTEGRATION.md](SYSTEM_PROMPT_INTEGRATION.md)** - System prompt feature
- **[DATABASE_MEMORY_MONITORING.md](DATABASE_MEMORY_MONITORING.md)** - DB/memory monitoring

---

## ✅ Summary

**What You Can Now Do:**
- ✅ View all 18 ontology classes
- ✅ Explore 15+ real system entities
- ✅ Understand 9 semantic relationships
- ✅ Study 17 inference rules
- ✅ See 3 complete reasoning examples
- ✅ Understand WHY Dheera makes decisions
- ✅ Learn HOW components interact
- ✅ See WHAT CONSTRAINTS apply

**Key Insight:**
This is NOT just a diagram. It's the **semantic blueprint** that makes Dheera a reasoning agent, not just autocomplete.

**Without ontology:** Dheera is if/else statements
**With ontology:** Dheera infers, reasons, explains

---

🧠 **The ontology is the brain. The code is the execution.**

Restart Streamlit and explore! 🎉
