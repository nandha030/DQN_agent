#!/usr/bin/env python3
# run_demo.py
"""
Dheera v0.3.0 - Demo Script
Demonstrates key features without user interaction.

Usage:
    python run_demo.py [--full]
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_header(title: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 50}")
    print(f"  {title}")
    print(f"{'═' * 50}{Colors.RESET}\n")


def print_step(step: str):
    """Print step indicator."""
    print(f"{Colors.YELLOW}▶ {step}{Colors.RESET}")


def print_success(msg: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")


def print_info(msg: str):
    """Print info message."""
    print(f"{Colors.DIM}  {msg}{Colors.RESET}")


def demo_component_tests():
    """Test individual components."""
    print_header("Component Tests")
    
    # Test embedding model
    print_step("Testing Embedding Model...")
    from rag.embeddings import EmbeddingModel
    emb_model = EmbeddingModel()
    emb = emb_model.embed("Hello world")
    print_success(f"Embedding shape: {emb.shape}")
    
    # Test intent classifier
    print_step("Testing Intent Classifier...")
    from cognitive.intent_classifier import IntentClassifier
    classifier = IntentClassifier()
    result = classifier.classify("What is Python?")
    print_success(f"Intent: {result.primary_intent.value} (confidence: {result.confidence:.2f})")
    
    # Test entity extractor
    print_step("Testing Entity Extractor...")
    from cognitive.entity_extractor import EntityExtractor
    extractor = EntityExtractor()
    result = extractor.extract("Send email to john@example.com about the meeting")
    entities = [f"{e.type.value}:{e.text}" for e in result.entities]
    print_success(f"Entities: {entities[:3]}")
    
    # Test reasoning engine
    print_step("Testing Reasoning Engine...")
    from cognitive.reasoning import ReasoningEngine
    reasoner = ReasoningEngine()
    result = reasoner.reason("Compare Python and JavaScript for web development")
    print_success(f"Reasoning type: {result.reasoning_type.value}")
    
    # Test state builder
    print_step("Testing State Builder...")
    from core.state_builder import StateBuilder
    builder = StateBuilder()
    state = builder.build_state("Hello, how are you?")
    print_success(f"State shape: {state.shape}")
    
    # Test action space
    print_step("Testing Action Space...")
    from core.action_space import ActionSpace
    action_space = ActionSpace()
    suggested = action_space.get_heuristic_action("Search for Python tutorials")
    print_success(f"Suggested action: {action_space.get_action_name(suggested or 0)}")


def demo_rainbow_dqn():
    """Demo Rainbow DQN."""
    print_header("Rainbow DQN Demo")
    
    import numpy as np
    from core.rainbow_dqn import RainbowDQNAgent
    
    print_step("Creating Rainbow DQN Agent...")
    agent = RainbowDQNAgent(
        state_dim=64,
        action_dim=8,
        hidden_dim=128,
    )
    print_success("Agent created")
    
    print_step("Testing action selection...")
    state = np.random.randn(64).astype(np.float32)
    action, info = agent.select_action(state)
    print_success(f"Selected action: {action}")
    print_info(f"Novelty score: {info.get('novelty', 0):.4f}")
    
    print_step("Storing transitions...")
    for i in range(100):
        s = np.random.randn(64).astype(np.float32)
        a = np.random.randint(0, 8)
        r = np.random.randn() * 0.5
        ns = np.random.randn(64).astype(np.float32)
        agent.store_transition(s, a, r, ns, False)
    print_success(f"Stored 100 transitions")
    
    print_step("Training step...")
    stats = agent.train_step()
    if stats:
        print_success(f"Training loss: {stats.get('loss', 0):.4f}")
    else:
        print_info("Not enough experiences for training yet")
    
    print_info(f"Total steps: {agent.total_steps}")


def demo_rag_system():
    """Demo RAG system."""
    print_header("RAG System Demo")
    
    import tempfile
    from rag.retriever import RAGRetriever
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print_step("Creating RAG Retriever...")
        rag = RAGRetriever(persist_directory=tmpdir)
        print_success("RAG system initialized")
        
        print_step("Adding documents...")
        rag.add_knowledge(
            "Python is a high-level programming language known for its simplicity.",
            source="knowledge"
        )
        rag.add_knowledge(
            "Machine learning is a subset of AI that learns from data.",
            source="knowledge"
        )
        rag.add_conversation_turn(
            turn_id="demo_1",
            user_message="What is Python?",
            assistant_response="Python is a versatile programming language.",
        )
        print_success(f"Added documents: {rag.get_stats()['total_documents']}")
        
        print_step("Retrieving similar documents...")
        result = rag.retrieve("Tell me about Python programming")
        print_success(f"Retrieved {result.total_found} documents")
        for doc in result.documents[:2]:
            print_info(f"  [{doc.metadata.get('source', 'unknown')}] {doc.text[:50]}...")


def demo_rlhf_system():
    """Demo RLHF system."""
    print_header("RLHF System Demo")
    
    import numpy as np
    from rlhf.reward_model import RewardModel
    from rlhf.preference_learner import PreferenceLearner, PreferencePair
    
    print_step("Creating Reward Model...")
    reward_model = RewardModel(state_dim=64, action_dim=8, response_dim=384)
    print_success("Reward model created")
    
    print_step("Predicting reward...")
    state = np.random.randn(64).astype(np.float32)
    response_emb = np.random.randn(384).astype(np.float32)
    reward = reward_model.predict_reward(state, 0, response_emb)
    print_success(f"Predicted reward: {reward:.4f}")
    
    print_step("Creating Preference Learner...")
    learner = PreferenceLearner(reward_model)
    
    print_step("Adding preference pairs...")
    for i in range(10):
        pref = PreferencePair(
            id=f"demo_{i}",
            state=np.random.randn(64).astype(np.float32),
            action_chosen=0,
            response_chosen_emb=np.random.randn(384).astype(np.float32),
            response_chosen_text="Good response",
            action_rejected=1,
            response_rejected_emb=np.random.randn(384).astype(np.float32),
            response_rejected_text="Bad response",
        )
        learner.add_preference(pref)
    print_success(f"Added {learner.total_preferences} preferences")
    
    print_step("Training on preferences...")
    stats = learner.train_step(batch_size=8)
    print_success(f"Training accuracy: {stats.get('accuracy', 0):.1%}")


def demo_full_system():
    """Demo full Dheera system."""
    print_header("Full System Demo")
    
    from dheera import Dheera
    
    print_step("Initializing Dheera...")
    dheera = Dheera()
    print_success(f"Dheera v{dheera.VERSION} ready")
    
    print_step("Starting conversation episode...")
    dheera.start_episode()
    
    # Demo conversations
    conversations = [
        "Hello! How are you today?",
        "What is machine learning?",
        "++",  # Positive feedback
        "Help me write a Python function",
        "Thanks, that was helpful!",
    ]
    
    for msg in conversations:
        print(f"\n{Colors.CYAN}User: {msg}{Colors.RESET}")
        response, metadata = dheera.process_message(msg)
        print(f"{Colors.GREEN}Dheera: {response[:150]}...{Colors.RESET}")
        print_info(f"Action: {metadata.get('action_name', 'N/A')}, "
                  f"Reward: {metadata.get('reward', 0):.3f}")
        time.sleep(0.1)
    
    print_step("Ending episode...")
    dheera.end_episode("Demo completed")
    
    print_step("Final Statistics:")
    stats = dheera.get_stats()
    print_info(f"DQN steps: {stats['dqn']['total_steps']}")
    print_info(f"RAG documents: {stats['rag']['total_documents']}")
    print_info(f"RLHF feedback: {stats['rlhf']['total_feedback']}")


def main():
    """Run demo."""
    parser = argparse.ArgumentParser(
        description="Dheera v0.3.0 - Demo Script"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full system demo (slower)"
    )
    parser.add_argument(
        "--component",
        choices=["embedding", "cognitive", "dqn", "rag", "rlhf", "all"],
        default="all",
        help="Which component to demo"
    )
    
    args = parser.parse_args()
    
    print(f"""
{Colors.CYAN}{Colors.BOLD}
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║      धीर  DHEERA v0.3.0 - DEMO                       ║
    ║      Courageous • Wise • Patient                      ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
{Colors.RESET}""")
    
    try:
        if args.component in ["all", "embedding", "cognitive"]:
            demo_component_tests()
        
        if args.component in ["all", "dqn"]:
            demo_rainbow_dqn()
        
        if args.component in ["all", "rag"]:
            demo_rag_system()
        
        if args.component in ["all", "rlhf"]:
            demo_rlhf_system()
        
        if args.full:
            demo_full_system()
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ Demo completed successfully!{Colors.RESET}\n")
        
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
