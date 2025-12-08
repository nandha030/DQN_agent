#!/usr/bin/env python3
"""
Dheera Demo Runner
A simple demo that works without an SLM connection.
Shows how the DQN learns from interactions.
"""

import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.dqn_agent import DQNAgent
from core.action_space import ActionSpace
from core.state_builder import StateBuilder, ConversationContext


def main():
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║       D H E E R A  -  DQN Demo                        ║
    ║       (No SLM required)                               ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # Initialize components
    action_space = ActionSpace()
    state_builder = StateBuilder()
    
    dqn = DQNAgent(
        state_dim=16,
        action_dim=action_space.size,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=500,
    )
    
    print(f"Actions available: {[a.name for a in action_space.actions]}\n")
    
    # Simulated conversations
    test_messages = [
        "What is machine learning?",
        "Can you explain it step by step?",
        "I'm confused about neural networks",
        "Write me a Python function",
        "Summarize this quickly",
        "Help me debug this code",
        "I don't understand, can you clarify?",
        "Perfect, thanks!",
        "This is too complicated",
        "Give me a detailed explanation",
    ]
    
    print("Simulating 100 interactions...\n")
    
    context = ConversationContext()
    prev_state = None
    prev_action = None
    
    for i in range(100):
        # Pick a random message
        msg = random.choice(test_messages)
        
        # Update context
        context.messages.append({"role": "user", "content": msg})
        context.turn_count += 1
        
        # Build state
        state = state_builder.build_state(context)
        
        # Select action
        action_id = dqn.select_action(state)
        action = action_space.get_by_id(action_id)
        
        # Simulate reward based on action appropriateness
        reward = 0.0
        if "explain" in msg.lower() and action.name == "PROVIDE_DETAILED_EXPLANATION":
            reward = 1.0
        elif "step" in msg.lower() and action.name == "PROPOSE_STEP_BY_STEP_PLAN":
            reward = 1.0
        elif "confused" in msg.lower() and action.name == "ASK_CLARIFYING_QUESTION":
            reward = 1.0
        elif "summarize" in msg.lower() and action.name == "GIVE_CONCISE_SUMMARY":
            reward = 1.0
        elif "code" in msg.lower() or "debug" in msg.lower():
            if action.name in ["REFLECT_AND_REASON", "PROPOSE_STEP_BY_STEP_PLAN"]:
                reward = 0.5
        elif "thanks" in msg.lower() or "perfect" in msg.lower():
            reward = 1.0
        else:
            reward = random.uniform(-0.2, 0.3)
        
        # Store transition
        if prev_state is not None:
            dqn.store_transition(prev_state, prev_action, reward, state, False)
        
        # Train
        train_info = dqn.train_step()
        
        # Log periodically
        if (i + 1) % 20 == 0:
            print(f"Step {i+1:3d} | Action: {action.name:30s} | Reward: {reward:+.2f} | Epsilon: {dqn.epsilon:.3f}")
        
        prev_state = state
        prev_action = action_id
        
        # Add simulated response to context
        context.messages.append({
            "role": "assistant",
            "content": f"[{action.name}] Simulated response..."
        })
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    # Test learned behavior
    print("\nTesting learned behavior:\n")
    
    test_queries = [
        "Can you explain quantum computing?",
        "Give me a quick summary",
        "I'm really confused about this",
        "Write a sorting algorithm",
    ]
    
    for query in test_queries:
        context = ConversationContext(
            messages=[{"role": "user", "content": query}],
            turn_count=1,
        )
        state = state_builder.build_state(context)
        
        # Get action (no exploration)
        action_id = dqn.select_action(state, explore=False)
        action = action_space.get_by_id(action_id)
        
        q_values = dqn.get_q_values(state)
        
        print(f"Query: \"{query}\"")
        print(f"  -> Action: {action.name}")
        q_dict = dict(zip([a.name[:10] for a in action_space.actions], [f"{q:.2f}" for q in q_values]))
        print(f"  -> Q-values: {q_dict}")
        print()
    
    print("\nDemo complete! The DQN has learned to match actions to queries.\n")


if __name__ == "__main__":
    main()
