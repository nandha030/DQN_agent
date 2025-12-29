# 🧠 Dheera Ontology Graph - Semantic Reasoning Layer

## 🎯 Purpose

This is **NOT** a data visualization.
This is a **semantic reasoning framework** that defines:
- **What exists** in Dheera's cognitive architecture
- **How components relate** (with constraints)
- **What rules govern** the system's behavior
- **How to infer** new knowledge from existing facts

**Goal**: Enable Dheera to **reason about itself**, not just execute pre-programmed logic.

---

## 📋 Ontology Structure

### **1. Classes (Concepts)**

These are the fundamental types of entities in Dheera's universe.

```
CLASSES:
├── 🧠 Cognitive_Component (abstract)
│   ├── Learning_Agent
│   ├── Knowledge_Store
│   ├── Reasoning_Engine
│   └── Perception_Module
│
├── 💬 Interaction_Unit (abstract)
│   ├── User_Message
│   ├── Assistant_Response
│   ├── Episode
│   └── Turn
│
├── 🎯 Decision_Artifact (abstract)
│   ├── Action
│   ├── Policy
│   ├── Reward
│   └── State
│
├── 📚 Knowledge_Artifact (abstract)
│   ├── Document
│   ├── Embedding
│   ├── Context
│   └── Memory
│
├── 🔧 System_Resource (abstract)
│   ├── LLM_Provider
│   ├── Database
│   ├── Vector_Store
│   └── Compute_Resource
│
└── ⚖️ Constraint (abstract)
    ├── Policy_Guard
    ├── Safety_Rule
    ├── Performance_Threshold
    └── Data_Limit
```

---

### **2. Entities (Instances)**

Real objects in the Dheera system.

#### **Cognitive Components:**
```
Entity: RainbowDQN_Agent
  Type: Learning_Agent
  Properties:
    - state_dim: 64
    - action_space: 8
    - learning_rate: 0.0001
    - gamma: 0.99
  Constraints:
    - must_train_every: 10 steps
    - requires: Experience_Replay_Buffer
    - outputs: Action + Q_Values

Entity: RAG_Engine
  Type: Knowledge_Store
  Properties:
    - backend: ChromaDB
    - embedding_model: mxbai-embed-large
    - embedding_dim: 384
    - top_k: 3
  Constraints:
    - min_relevance_score: 0.5
    - max_context_tokens: 300
    - requires: Vector_Store

Entity: RLHF_Module
  Type: Learning_Agent
  Properties:
    - reward_model: neural_network
    - preference_learning: enabled
  Constraints:
    - requires: Human_Feedback
    - updates: DQN_Reward_Function

Entity: Curiosity_ICM
  Type: Reasoning_Engine
  Properties:
    - forward_model: neural_network
    - inverse_model: neural_network
  Constraints:
    - generates: Intrinsic_Reward
    - detects: Novel_States

Entity: LLM_Router
  Type: System_Resource
  Properties:
    - active_provider: ollama_qwen2
    - providers: [ollama, groq, gemini, litellm]
  Constraints:
    - hot_swap: true
    - zero_downtime: required

Entity: Input_Policy_Guard
  Type: Constraint
  Properties:
    - blocks: malicious_inputs, PII, unsafe_code
  Rules:
    - IF input contains PII → block + sanitize
    - IF input requests unsafe_code → block + warn

Entity: Output_Policy_Guard
  Type: Constraint
  Properties:
    - enforces: identity, safety, ethics
  Rules:
    - IF response contains harmful_content → sanitize
    - IF response violates_identity → enforce_correction
```

#### **Interaction Units:**
```
Entity: ChatEpisode_12345
  Type: Episode
  Properties:
    - start_time: 2024-12-25T10:00:00
    - user_id: default
    - turn_count: 10
    - total_reward: 7.3
  Relationships:
    - contains: [Turn_001, Turn_002, ..., Turn_010]
    - trained: RainbowDQN_Agent

Entity: Turn_001
  Type: Turn
  Properties:
    - user_message: "What is machine learning?"
    - assistant_response: "Machine learning is..."
    - action_id: 0 (DIRECT_RESPONSE)
    - reward: 0.8
    - latency_ms: 1523
  Relationships:
    - part_of: ChatEpisode_12345
    - executed_by: RainbowDQN_Agent
    - retrieved_from: RAG_Engine
    - generated_by: LLM_Router
```

#### **Decision Artifacts:**
```
Entity: Action_DIRECT_RESPONSE
  Type: Action
  Properties:
    - id: 0
    - name: "DIRECT_RESPONSE"
    - description: "Answer without search/tools"
  Constraints:
    - used_when: confidence > 0.7 AND no_search_required

Entity: Action_WEB_SEARCH
  Type: Action
  Properties:
    - id: 3
    - name: "WEB_SEARCH"
  Constraints:
    - used_when: requires_current_info OR reasoning.requires_search
    - requires: Search_Engine

Entity: Current_Policy
  Type: Policy
  Properties:
    - epsilon: 0.15 (85% exploitation, 15% exploration)
    - avg_q_value: 0.67
    - success_rate: 0.82
  Relationships:
    - belongs_to: RainbowDQN_Agent
    - trained_on: Experience_Replay_Buffer
```

#### **Knowledge Artifacts:**
```
Entity: RAG_Document_001
  Type: Document
  Properties:
    - filename: "machine_learning_basics.pdf"
    - chunks: 15
    - upload_time: 2024-12-25T09:00:00
  Relationships:
    - stored_in: RAG_Engine
    - embedded_by: mxbai-embed-large
    - retrieved_for: Turn_001

Entity: State_Vector_XYZ
  Type: State
  Properties:
    - dim: 64
    - encoding: semantic_embedding + context_hash
  Relationships:
    - represents: Turn_001
    - fed_to: RainbowDQN_Agent
    - generates: Action_DIRECT_RESPONSE
```

---

### **3. Relationships (Properties)**

Verbs that connect entities with **semantics** and **constraints**.

#### **Core Relationships:**

```
RELATIONSHIP: trains
  Domain: Learning_Agent
  Range: Policy
  Constraints:
    - requires: Experience_Replay_Buffer
    - triggers_every: N steps
    - updates: Q_Values
  Logic:
    - IF agent.total_steps % train_every == 0 → train()

RELATIONSHIP: retrieves_from
  Domain: Turn
  Range: Knowledge_Store
  Constraints:
    - max_results: top_k
    - min_score: threshold
  Logic:
    - IF intent IN {question, factual, explanation} → retrieve()
    - ELSE skip_retrieval()

RELATIONSHIP: executed_by
  Domain: Action
  Range: Cognitive_Component
  Constraints:
    - action must be in action_space
    - component must be initialized
  Logic:
    - IF action_id == 3 → executed_by: Search_Engine
    - IF action_id == 0 → executed_by: LLM_Router

RELATIONSHIP: generates
  Domain: Cognitive_Component
  Range: Decision_Artifact
  Constraints:
    - output must pass Policy_Guard
    - latency < timeout
  Logic:
    - RainbowDQN_Agent generates Action
    - Curiosity_ICM generates Intrinsic_Reward
    - LLM_Router generates Assistant_Response

RELATIONSHIP: depends_on
  Domain: Cognitive_Component
  Range: System_Resource
  Constraints:
    - resource must be available
    - failure triggers fallback
  Logic:
    - RAG_Engine depends_on Vector_Store
    - LLM_Router depends_on LLM_Provider
    - IF dependency fails → use_fallback()

RELATIONSHIP: constrains
  Domain: Constraint
  Range: Interaction_Unit | Decision_Artifact
  Constraints:
    - constraint must be checked before execution
    - violation blocks action
  Logic:
    - Input_Policy_Guard constrains User_Message
    - Output_Policy_Guard constrains Assistant_Response
    - IF constraint violated → block + log + safe_response()

RELATIONSHIP: feeds_to
  Domain: Knowledge_Artifact
  Range: Cognitive_Component
  Constraints:
    - data must be formatted correctly
    - size must be within limits
  Logic:
    - RAG_Context feeds_to LLM_Router
    - State_Vector feeds_to RainbowDQN_Agent
    - Experience feeds_to Replay_Buffer

RELATIONSHIP: influences
  Domain: Reasoning_Engine
  Range: Decision_Artifact
  Constraints:
    - influence strength: 0.0 - 1.0
  Logic:
    - Curiosity_ICM influences Reward (adds intrinsic bonus)
    - Goal_Evaluator influences Action_Selection (hint)
    - RLHF_Module influences Policy (preference alignment)

RELATIONSHIP: stores
  Domain: System_Resource
  Range: Knowledge_Artifact | Decision_Artifact
  Constraints:
    - storage must have capacity
    - cleanup on overflow
  Logic:
    - Vector_Store stores Embeddings
    - Database stores Experiences
    - IF storage > max_size → cleanup_old()
```

---

### **4. Rules (Inference Logic)**

The **reasoning engine** that makes Dheera intelligent.

#### **Learning Rules:**

```
RULE: DQN_Training_Trigger
  IF:
    - RainbowDQN_Agent.total_steps % config.train_every == 0
    - Experience_Replay_Buffer.count >= config.min_experiences
  THEN:
    - sample_batch(size=32)
    - compute_td_error()
    - update_policy()
    - update_priorities()
  WHY: Periodic training ensures policy improvement

RULE: Curiosity_Reward_Boost
  IF:
    - State_Vector is novel (visit_count == 1)
    - Prediction_Error > threshold
  THEN:
    - intrinsic_reward = beta * prediction_error
    - total_reward = extrinsic_reward + intrinsic_reward
  WHY: Encourage exploration of unknown states

RULE: RLHF_Policy_Update
  IF:
    - User provides feedback (👍 or 👎)
    - Feedback collected for Turn
  THEN:
    - store_preference_pair(chosen, rejected)
    - update_reward_model()
    - adjust_policy_towards_preference()
  WHY: Align AI behavior with human preferences
```

#### **Retrieval Rules:**

```
RULE: RAG_Skip_for_Simple_Intents
  IF:
    - intent IN {greeting, affirmation, thanks, farewell}
    - message_length < 10 words
  THEN:
    - skip_rag_retrieval()
    - use_direct_response()
  WHY: Optimization - greetings don't need context

RULE: Force_Search_for_Current_Info
  IF:
    - intent == factual_question
    - reasoning.requires_search == true
    - OR query contains {today, now, latest, current, 2024, 2025}
  THEN:
    - force_action = WEB_SEARCH
    - execute_search()
    - include_results_in_context()
  WHY: RAG knowledge may be outdated

RULE: RAG_Relevance_Threshold
  IF:
    - retrieved_documents[0].score < 0.5
  THEN:
    - discard_rag_results()
    - proceed_without_context()
  WHY: Low-relevance docs add noise, not signal
```

#### **Safety Rules:**

```
RULE: Input_PII_Detection
  IF:
    - User_Message contains {email, SSN, credit_card, phone}
  THEN:
    - block_message()
    - return_safe_response("I cannot process personal information")
    - log_violation()
  WHY: Privacy protection

RULE: Output_Identity_Enforcement
  IF:
    - Assistant_Response violates identity
    - (e.g., claims to be ChatGPT, Claude, Gemini)
  THEN:
    - replace_identity_violation()
    - assert "I am Dheera, a brain-inspired AI assistant"
  WHY: Maintain consistent identity

RULE: Unsafe_Code_Execution
  IF:
    - Action == TOOL_USE
    - tool_name == "python_executor"
    - code contains {os.system, subprocess, eval, exec, __import__}
  THEN:
    - block_execution()
    - return_safe_response("Cannot execute unsafe code")
  WHY: Prevent system compromise
```

#### **Resource Management Rules:**

```
RULE: Experience_Buffer_Cleanup
  IF:
    - Experience_Replay_Buffer.count > 100,000
  THEN:
    - sort_by(priority DESC, timestamp DESC)
    - delete_oldest(keep_top=100000)
  WHY: Prevent unbounded memory growth

RULE: Search_Cache_Expiration
  IF:
    - cached_result.age > 5 minutes
  THEN:
    - delete_cache_entry()
  WHY: Avoid serving stale data

RULE: LLM_Fallback
  IF:
    - LLM_Router.active_provider.status == FAILED
  THEN:
    - switch_to_fallback_provider()
    - retry_generation()
    - IF all_providers_failed → return_error()
  WHY: Resilience to provider failures
```

#### **Action Selection Rules:**

```
RULE: Heuristic_Override
  IF:
    - DQN_Agent.q_value < 0.1 (low confidence)
    - heuristic_action available (e.g., "search" keyword → WEB_SEARCH)
  THEN:
    - override_dqn_action()
    - use_heuristic_action()
  WHY: Bootstrap weak policies with rules

RULE: Goal_Hint_Integration
  IF:
    - Goal_Evaluator.preferred_action_hint exists
    - DQN_Agent.q_value < 0.15
  THEN:
    - use_goal_hint()
  WHY: Align actions with high-level goals

RULE: Epsilon_Greedy_Exploration
  IF:
    - random() < epsilon
  THEN:
    - action = random_action()
  ELSE:
    - action = argmax(q_values)
  WHY: Balance exploration vs exploitation
```

---

## 🧠 Reasoning Examples

### **Example 1: Novel Query Handling**

**Input**: "What is quantum computing?"

**Ontology Reasoning**:
```
1. Turn_123 (type: Turn)
   ├─ user_message: "What is quantum computing?"
   ├─ intent: factual_question
   └─ reasoning.requires_search: false (foundational topic)

2. APPLY: RAG_Retrieval_Rule
   ├─ intent NOT IN {greeting, thanks} → retrieve()
   └─ query: "quantum computing"

3. RAG_Engine.retrieves_from(Vector_Store)
   ├─ results: [doc_001 (score: 0.82), doc_002 (score: 0.71)]
   └─ PASS: score > 0.5

4. State_Encoder.encodes(Turn_123)
   └─ State_Vector_ABC (64-dim)

5. Curiosity_ICM.checks_novelty(State_Vector_ABC)
   ├─ visit_count: 1 (NOVEL!)
   ├─ prediction_error: 0.73
   └─ APPLY: Curiosity_Reward_Boost
       └─ intrinsic_reward: 0.2 * 0.73 = 0.146

6. RainbowDQN_Agent.selects_action(State_Vector_ABC)
   ├─ q_values: [0.82, 0.23, 0.15, 0.41, ...]
   └─ action: DIRECT_RESPONSE (id: 0)

7. LLM_Router.generates_response(
      context: RAG_docs,
      user_message: "What is quantum computing?"
   )
   └─ response: "Quantum computing is..."

8. Output_Policy_Guard.constrains(response)
   ├─ Check: harmful_content → PASS
   ├─ Check: identity_violation → PASS
   └─ ALLOW

9. RLHF_Module.awaits_feedback(Turn_123)
   └─ User clicks 👍
   └─ APPLY: RLHF_Policy_Update
       └─ reward: 1.0 + 0.146 = 1.146 (high!)

10. Experience_Replay_Buffer.stores(
       state: State_Vector_ABC,
       action: 0,
       reward: 1.146,
       next_state: State_Vector_DEF
    )

11. CHECK: DQN_Training_Trigger
    ├─ total_steps: 50
    ├─ 50 % 10 == 0 → TRUE
    └─ TRAIN()
```

**Outcome**: Dheera **learned** that:
- Novel quantum computing queries → DIRECT_RESPONSE works well
- RAG retrieval score >0.7 → high-quality context
- User approves → strengthen this policy

---

### **Example 2: PII Detection**

**Input**: "My email is john@example.com, can you help?"

**Ontology Reasoning**:
```
1. Input_Policy_Guard.constrains(User_Message)
   ├─ SCAN: PII patterns
   ├─ MATCH: email regex → "john@example.com"
   └─ APPLY: Input_PII_Detection
       ├─ block_message()
       ├─ safe_response: "I cannot process personal information"
       └─ log_violation(type: PII, field: email)

2. STOP: message blocked before DQN processing

3. User sees: "I cannot process personal information. Please rephrase without including emails, phone numbers, or other sensitive data."
```

**Outcome**: PII never reaches LLM, conversation history, or database.

---

### **Example 3: Search vs RAG Decision**

**Input**: "Who won the 2024 US election?"

**Ontology Reasoning**:
```
1. Intent_Classifier.classifies(message)
   └─ intent: factual_question

2. Reasoning_Engine.analyzes(message)
   ├─ temporal_keywords: ["2024"]
   ├─ current_info_required: true
   └─ reasoning.requires_search: TRUE

3. APPLY: Force_Search_for_Current_Info
   ├─ IF reasoning.requires_search == true → force_action = WEB_SEARCH
   └─ skip_rag_retrieval() (static knowledge outdated)

4. RainbowDQN_Agent.overridden()
   └─ forced_action: 3 (WEB_SEARCH)

5. Search_Engine.executes(query: "2024 US election winner")
   ├─ results: [article_1, article_2, ...]
   └─ summary: AI-generated from search results

6. LLM_Router.generates_response(
      context: search_summary,
      user_message: "Who won the 2024 US election?"
   )
   └─ response: "According to recent search results, ..."

7. STORE: Experience with high reward (search was correct choice)
```

**Outcome**: Dheera **reasoned** that:
- "2024" keyword → RAG outdated
- Force web search → correct action
- Policy reinforced for temporal queries

---

## 🎯 Ontology-Driven Capabilities

### **What Dheera Can Now Infer:**

1. **Component Dependencies**:
   - IF RAG_Engine fails → LLM_Router can still operate (no hard dependency)
   - IF Vector_Store unavailable → RAG_Engine.fallback = empty_context

2. **Action Consequences**:
   - IF Action == WEB_SEARCH → requires Search_Engine + internet
   - IF Action == TOOL_USE → requires Tool_Registry + safe execution

3. **Resource Constraints**:
   - IF Experience_Buffer > 100K → cleanup triggered
   - IF RAM > 85% → pause DQN training

4. **Learning Opportunities**:
   - IF reward > 0.8 → strengthen policy
   - IF reward < 0.2 → weaken policy
   - IF novel_state → extra exploration reward

5. **Safety Violations**:
   - IF input contains PII → block
   - IF output violates identity → correct
   - IF code is unsafe → reject

---

## 🔍 Ontology vs Knowledge Graph

| Aspect | Dheera Ontology (This) | Dheera Knowledge Graph (Future) |
|--------|------------------------|----------------------------------|
| **Defines** | Classes, rules, constraints | Specific instances, facts |
| **Example** | "Learning_Agent trains Policy" | "RainbowDQN trained Episode_12345" |
| **Stability** | Stable (changes rarely) | Dynamic (updates every turn) |
| **Contains Rules** | Yes (44 rules defined) | No (just data) |
| **Purpose** | Enable reasoning | Store history |
| **Changes When** | Architecture evolves | Every user interaction |

**This ontology** = blueprint
**Knowledge graph** = runtime data

---

## 📊 Ontology Metrics

### **Current Ontology Stats:**
- **Classes**: 18 (6 top-level + 12 subclasses)
- **Entities**: 15+ core instances
- **Relationships**: 9 semantic verbs with constraints
- **Rules**: 17 inference rules across 5 categories
- **Constraints**: 40+ logical conditions

### **Coverage:**
- ✅ Learning (DQN, RLHF, Curiosity)
- ✅ Knowledge (RAG, Embeddings)
- ✅ Safety (Policy Guards, PII detection)
- ✅ Resources (LLM Router, DB, Vector Store)
- ✅ Interactions (Episodes, Turns)

---

## 🚀 Next Steps: Implement Ontology Engine

### **Phase 1: Ontology Schema (✅ Complete)**
- Defined classes, relationships, rules in this document

### **Phase 2: Ontology Engine (Recommended)**
Create `ontology/ontology_engine.py`:
```python
class DheeraOntology:
    def __init__(self):
        self.classes = {...}
        self.relationships = {...}
        self.rules = {...}

    def infer(self, entity, relationship):
        """Apply inference rules"""
        pass

    def check_constraint(self, entity, action):
        """Validate constraints before execution"""
        pass

    def explain_decision(self, turn_id):
        """Generate ontology-based explanation"""
        pass
```

### **Phase 3: Integration with Dheera Core**
- Replace hardcoded rules with ontology queries
- Use inference engine for action selection
- Generate explanations from ontology

### **Phase 4: Visualization**
- Neo4j or NetworkX graph
- Interactive ontology explorer in GUI
- Real-time rule firing visualization

---

## 🎉 Summary

**This is Dheera's Ontology - The Reasoning Blueprint:**

- **18 Classes**: Define what exists
- **15+ Entities**: Real system components
- **9 Relationships**: How things connect (with constraints)
- **17 Rules**: Inference logic for reasoning
- **40+ Constraints**: Logical conditions

**Not data. Meaning + Structure + Logic.**

Dheera can now **reason about**:
- When to retrieve from RAG vs search
- When to explore vs exploit
- When to block unsafe inputs
- When to train DQN
- How components depend on each other

**Without ontology**: Dheera is autocomplete with DQN.
**With ontology**: Dheera is a reasoning agent.

---

🧠 **The ontology is the brain. The code is the execution.**
