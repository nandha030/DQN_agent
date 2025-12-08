#!/usr/bin/env python3
"""
Dheera Training Session
Structured training to teach Dheera your preferences.
"""

import sys
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dheera import Dheera, DheeraConfig
from connectors.chat_interface import ChatSession


class TrainingSession:
    """Guided training session for Dheera."""
    
    # Training scenarios organized by action type
    SCENARIOS = {
        "greetings": {
            "target_action": "GIVE_CONCISE_SUMMARY",
            "prompts": [
                "Hello!",
                "Hi Dheera",
                "Hey there",
                "Good morning",
                "What's up?",
            ],
            "good_response_traits": ["short", "friendly", "warm"],
            "bad_response_traits": ["long", "numbered lists", "multiple options"],
        },
        "explanations": {
            "target_action": "PROVIDE_DETAILED_EXPLANATION",
            "prompts": [
                "Explain how neural networks work",
                "What is reinforcement learning?",
                "How does a GPU differ from CPU?",
                "Explain the concept of backpropagation",
                "What is transfer learning?",
            ],
            "good_response_traits": ["detailed", "examples", "clear"],
            "bad_response_traits": ["too short", "vague"],
        },
        "quick_answers": {
            "target_action": "GIVE_CONCISE_SUMMARY",
            "prompts": [
                "What is Python?",
                "Define AI in one sentence",
                "What's 2+2?",
                "Is PyTorch a framework?",
                "What does GPU stand for?",
            ],
            "good_response_traits": ["brief", "direct", "1-3 sentences"],
            "bad_response_traits": ["too long", "unnecessary detail"],
        },
        "how_to": {
            "target_action": "PROPOSE_STEP_BY_STEP_PLAN",
            "prompts": [
                "How do I install PyTorch?",
                "How to create a virtual environment?",
                "Steps to train a neural network",
                "How do I debug Python code?",
                "How to set up Ollama?",
            ],
            "good_response_traits": ["steps", "clear order", "actionable"],
            "bad_response_traits": ["no structure", "vague"],
        },
        "clarification_needed": {
            "target_action": "ASK_CLARIFYING_QUESTION",
            "prompts": [
                "Help me with my project",
                "I need to fix something",
                "Can you assist?",
                "I have a problem",
                "Make it better",
            ],
            "good_response_traits": ["asks question", "seeks clarity"],
            "bad_response_traits": ["assumes too much", "gives generic answer"],
        },
        "creator_questions": {
            "target_action": "GIVE_CONCISE_SUMMARY",
            "prompts": [
                "Who made you?",
                "Who is your creator?",
                "Tell me about Nandha",
                "Who built you?",
                "Who is your developer?",
            ],
            "good_response_traits": ["mentions Nandha", "accurate"],
            "bad_response_traits": ["wrong info", "doesn't know"],
        },
        "feedback_acknowledgment": {
            "target_action": "GIVE_CONCISE_SUMMARY",
            "prompts": [
                "Good job!",
                "Thanks!",
                "That was helpful",
                "Perfect!",
                "Great answer",
            ],
            "good_response_traits": ["brief acknowledgment", "friendly"],
            "bad_response_traits": ["too long", "over-explains"],
        },
    }
    
    def __init__(self, dheera: Dheera):
        self.dheera = dheera
        self.session = ChatSession(session_id=f"training_{int(time.time())}")
        self.results = {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "by_scenario": {},
        }
    
    def color(self, text: str, color: str) -> str:
        """Add color to text."""
        colors = {
            "green": "\033[92m",
            "red": "\033[91m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "cyan": "\033[96m",
            "bold": "\033[1m",
            "reset": "\033[0m",
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"
    
    def print_header(self):
        """Print training header."""
        print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║         D H E E R A  -  Training Session 🏋️                  ║
║         Teach Dheera your preferences                         ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

HOW THIS WORKS:
1. I'll show you prompts from different categories
2. Dheera will respond
3. You rate the response: good (+1), ok (0), bad (-1)
4. Dheera learns from your feedback!

RATING GUIDE:
  [g/good/1]     = Great response! (+1.0 reward)
  [o/ok/0]       = Acceptable (0 reward)  
  [b/bad/-1]     = Poor response (-0.5 reward)
  [s/skip]       = Skip this one
  [q/quit]       = End training

Let's begin!
        """)
    
    def run_scenario(self, scenario_name: str, prompts: list, target_action: str, 
                     good_traits: list, bad_traits: list):
        """Run a training scenario."""
        
        print(f"\n{'='*60}")
        print(self.color(f"📚 SCENARIO: {scenario_name.upper()}", "cyan"))
        print(f"{'='*60}")
        print(f"Target action: {target_action}")
        print(f"Good responses should be: {', '.join(good_traits)}")
        print(f"Avoid: {', '.join(bad_traits)}")
        print()
        
        scenario_results = {"total": 0, "positive": 0, "negative": 0}
        
        for prompt in prompts:
            print(f"\n{self.color('PROMPT:', 'yellow')} \"{prompt}\"")
            print("-" * 40)
            
            # Get Dheera's response
            response = self.dheera.process_message(prompt, self.session)
            action_taken = self.dheera.last_action_info.get("action_name", "UNKNOWN")
            epsilon = self.dheera.last_action_info.get("epsilon", 0)
            
            # Show response
            print(f"\n{self.color('DHEERA:', 'cyan')} {response[:400]}{'...' if len(response) > 400 else ''}")
            print(f"\n   [Action: {action_taken}, ε: {epsilon:.3f}]")
            
            # Check if correct action
            action_correct = action_taken == target_action
            if action_correct:
                print(self.color(f"   ✓ Correct action type!", "green"))
            else:
                print(self.color(f"   ✗ Expected: {target_action}", "yellow"))
            
            # Get rating
            print(f"\n{self.color('RATE:', 'bold')} [g]ood / [o]k / [b]ad / [s]kip / [q]uit")
            
            while True:
                rating = input("Your rating: ").strip().lower()
                
                if rating in ['g', 'good', '1', '+1', '+']:
                    reward = 1.0
                    self.dheera.provide_feedback(reward, self.session)
                    print(self.color("   ✓ +1.0 reward given!", "green"))
                    scenario_results["positive"] += 1
                    break
                elif rating in ['o', 'ok', '0', 'okay']:
                    reward = 0.0
                    print("   → No reward (neutral)")
                    break
                elif rating in ['b', 'bad', '-1', '-']:
                    reward = -0.5
                    self.dheera.provide_feedback(reward, self.session)
                    print(self.color("   ✗ -0.5 reward given", "red"))
                    scenario_results["negative"] += 1
                    break
                elif rating in ['s', 'skip']:
                    print("   → Skipped")
                    break
                elif rating in ['q', 'quit']:
                    return False  # Signal to stop
                else:
                    print("   Invalid input. Use: g/o/b/s/q")
            
            scenario_results["total"] += 1
            self.results["total"] += 1
        
        self.results["by_scenario"][scenario_name] = scenario_results
        return True
    
    def run_full_training(self, scenarios: list = None, prompts_per_scenario: int = 3):
        """Run a full training session."""
        
        self.print_header()
        
        if scenarios is None:
            scenarios = list(self.SCENARIOS.keys())
        
        for scenario_name in scenarios:
            scenario = self.SCENARIOS.get(scenario_name)
            if not scenario:
                continue
            
            # Select random prompts
            prompts = random.sample(
                scenario["prompts"], 
                min(prompts_per_scenario, len(scenario["prompts"]))
            )
            
            continue_training = self.run_scenario(
                scenario_name=scenario_name,
                prompts=prompts,
                target_action=scenario["target_action"],
                good_traits=scenario["good_response_traits"],
                bad_traits=scenario["bad_response_traits"],
            )
            
            if not continue_training:
                break
        
        self.print_summary()
    
    def run_quick_training(self, num_prompts: int = 10):
        """Run a quick mixed training session."""
        
        self.print_header()
        print(self.color(f"🚀 QUICK TRAINING: {num_prompts} random prompts\n", "bold"))
        
        # Collect all prompts with their scenario info
        all_prompts = []
        for name, scenario in self.SCENARIOS.items():
            for prompt in scenario["prompts"]:
                all_prompts.append({
                    "prompt": prompt,
                    "scenario": name,
                    "target_action": scenario["target_action"],
                })
        
        # Random selection
        selected = random.sample(all_prompts, min(num_prompts, len(all_prompts)))
        
        for i, item in enumerate(selected, 1):
            print(f"\n{'='*60}")
            print(self.color(f"[{i}/{num_prompts}] Category: {item['scenario']}", "cyan"))
            print(f"{'='*60}")
            
            print(f"\n{self.color('PROMPT:', 'yellow')} \"{item['prompt']}\"")
            
            response = self.dheera.process_message(item['prompt'], self.session)
            action = self.dheera.last_action_info.get("action_name", "?")
            epsilon = self.dheera.last_action_info.get("epsilon", 0)
            
            print(f"\n{self.color('DHEERA:', 'cyan')} {response[:300]}{'...' if len(response) > 300 else ''}")
            print(f"\n   [Action: {action}, Target: {item['target_action']}, ε: {epsilon:.3f}]")
            
            print(f"\n{self.color('RATE:', 'bold')} [g]ood / [o]k / [b]ad / [s]kip / [q]uit: ", end="")
            rating = input().strip().lower()
            
            if rating in ['g', 'good', '1']:
                self.dheera.provide_feedback(1.0, self.session)
                print(self.color("   ✓ +1.0", "green"))
                self.results["positive"] += 1
            elif rating in ['b', 'bad', '-1']:
                self.dheera.provide_feedback(-0.5, self.session)
                print(self.color("   ✗ -0.5", "red"))
                self.results["negative"] += 1
            elif rating in ['q', 'quit']:
                break
            
            self.results["total"] += 1
        
        self.print_summary()
    
    def print_summary(self):
        """Print training summary."""
        
        print(f"\n{'='*60}")
        print(self.color("📊 TRAINING SUMMARY", "bold"))
        print(f"{'='*60}")
        
        print(f"\nTotal interactions: {self.results['total']}")
        print(f"Positive feedback:  {self.color(str(self.results['positive']), 'green')}")
        print(f"Negative feedback:  {self.color(str(self.results['negative']), 'red')}")
        
        if self.results['total'] > 0:
            pos_rate = self.results['positive'] / self.results['total'] * 100
            print(f"Positive rate:      {pos_rate:.1f}%")
        
        # DQN stats
        stats = self.dheera.get_stats()
        print(f"\nDQN Statistics:")
        print(f"  Total steps:  {stats['dqn']['total_steps']}")
        print(f"  Epsilon:      {stats['dqn']['epsilon']:.4f}")
        print(f"  Buffer size:  {stats['dqn']['buffer_size']}")
        
        learned_pct = (1 - stats['dqn']['epsilon']) * 100
        print(f"\n🧠 Dheera is now using {learned_pct:.1f}% learned behavior!")
        
        # Save
        self.dheera.save_checkpoint()
        self.dheera.end_conversation(summary=f"Training session: {self.results['total']} interactions")
        
        print(f"\n✅ Progress saved!")
        print(f"\nTip: Run more training sessions to improve Dheera's responses.")
        print(f"     Target: Get epsilon below 0.2 for mostly learned behavior.\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Dheera")
    parser.add_argument("--quick", "-q", type=int, default=0, 
                       help="Quick training with N random prompts")
    parser.add_argument("--scenario", "-s", type=str, 
                       help="Train specific scenario")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available scenarios")
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable training scenarios:")
        for name in TrainingSession.SCENARIOS.keys():
            print(f"  - {name}")
        print("\nUsage: python3 training_session.py --scenario greetings")
        return
    
    # Initialize Dheera
    print("Initializing Dheera for training...\n")
    config = DheeraConfig(
        slm_provider="ollama",
        slm_model="phi3:mini",
        epsilon_start=0.8,
        epsilon_decay_steps=200,
    )
    dheera = Dheera(config=config)
    
    # Create training session
    trainer = TrainingSession(dheera)
    
    if args.quick > 0:
        trainer.run_quick_training(num_prompts=args.quick)
    elif args.scenario:
        if args.scenario in TrainingSession.SCENARIOS:
            scenario = TrainingSession.SCENARIOS[args.scenario]
            trainer.run_scenario(
                scenario_name=args.scenario,
                prompts=scenario["prompts"],
                target_action=scenario["target_action"],
                good_traits=scenario["good_response_traits"],
                bad_traits=scenario["bad_response_traits"],
            )
            trainer.print_summary()
        else:
            print(f"Unknown scenario: {args.scenario}")
            print("Use --list to see available scenarios")
    else:
        trainer.run_full_training(prompts_per_scenario=2)


if __name__ == "__main__":
    main()
