#!/usr/bin/env python3
# training_session.py
"""
Dheera v0.3.0 - Training Session
Run automated training sessions with synthetic data.

Usage:
    python training_session.py [--episodes N] [--turns N] [--save]
"""

import os
import sys
import argparse
import random
import time
from typing import List, Dict, Tuple
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dheera import Dheera


# Training data - diverse conversation samples
TRAINING_CONVERSATIONS = [
    # Greetings
    ("Hello!", 0.3),
    ("Hi there, how are you?", 0.3),
    ("Good morning!", 0.3),
    ("Hey Dheera!", 0.4),
    
    # Simple questions
    ("What is Python?", 0.5),
    ("Who created Linux?", 0.5),
    ("What is machine learning?", 0.6),
    ("Define artificial intelligence", 0.5),
    
    # How-to questions
    ("How do I install Python?", 0.6),
    ("How to read a CSV file in Python?", 0.7),
    ("How can I learn programming?", 0.6),
    ("How do I create a virtual environment?", 0.7),
    
    # Code requests
    ("Write a function to calculate factorial", 0.8),
    ("Help me debug this Python code", 0.7),
    ("Create a simple web scraper", 0.8),
    ("Write a sorting algorithm", 0.7),
    
    # Search queries
    ("What's the latest news about AI?", 0.6),
    ("Search for Python tutorials", 0.5),
    ("Find information about climate change", 0.6),
    ("What's the weather like today?", 0.4),
    
    # Math questions
    ("Calculate 25 * 4 + 10", 0.9),
    ("What is 15% of 200?", 0.9),
    ("Solve: 2x + 5 = 15", 0.8),
    
    # Complex questions
    ("Compare Python and JavaScript", 0.7),
    ("Explain how neural networks work", 0.7),
    ("Why is the sky blue?", 0.6),
    ("What are the pros and cons of microservices?", 0.7),
    
    # Clarification needed
    ("Tell me about that", 0.3),
    ("What about the other thing?", 0.3),
    ("Can you explain more?", 0.4),
    
    # Follow-ups
    ("Can you give me an example?", 0.6),
    ("What else should I know?", 0.5),
    ("Thanks, that was helpful", 0.8),
]

# Feedback patterns
FEEDBACK_POSITIVE = ["++", "+"]
FEEDBACK_NEGATIVE = ["--", "-"]


class TrainingSession:
    """Automated training session for Dheera."""
    
    def __init__(
        self,
        config_path: str = "config/dheera_config.yaml",
        verbose: bool = True,
    ):
        self.verbose = verbose
        
        print("🏋️ Initializing Training Session...")
        self.dheera = Dheera(config_path=config_path)
        
        # Statistics
        self.total_turns = 0
        self.total_episodes = 0
        self.total_reward = 0.0
        self.feedback_given = 0
        self.start_time = None
    
    def run_episode(
        self,
        turns: int = 10,
        feedback_probability: float = 0.3,
    ) -> Dict:
        """
        Run a single training episode.
        
        Args:
            turns: Number of turns in episode
            feedback_probability: Chance of giving feedback
            
        Returns:
            Episode statistics
        """
        self.dheera.start_episode(user_id="trainer")
        
        episode_reward = 0.0
        episode_turns = 0
        
        # Select random conversations
        conversations = random.sample(
            TRAINING_CONVERSATIONS,
            min(turns, len(TRAINING_CONVERSATIONS))
        )
        
        for message, expected_quality in conversations:
            # Process message
            response, metadata = self.dheera.process_message(message)
            
            episode_reward += metadata.get('reward', 0)
            episode_turns += 1
            
            if self.verbose:
                action = metadata.get('action_name', 'N/A')
                reward = metadata.get('reward', 0)
                print(f"  [{action}] {message[:40]}... -> reward={reward:.2f}")
            
            # Maybe give feedback
            if random.random() < feedback_probability:
                # Decide feedback based on reward
                actual_reward = metadata.get('reward', 0)
                
                if actual_reward > expected_quality * 0.8:
                    # Good response
                    feedback = random.choice(FEEDBACK_POSITIVE)
                elif actual_reward < expected_quality * 0.3:
                    # Bad response
                    feedback = random.choice(FEEDBACK_NEGATIVE)
                else:
                    # Skip feedback
                    feedback = None
                
                if feedback:
                    self.dheera.process_message(feedback)
                    self.feedback_given += 1
                    if self.verbose:
                        print(f"    Feedback: {feedback}")
            
            # Small delay to simulate real conversation
            time.sleep(0.01)
        
        self.dheera.end_episode("Training episode completed")
        
        self.total_turns += episode_turns
        self.total_reward += episode_reward
        self.total_episodes += 1
        
        return {
            "turns": episode_turns,
            "reward": episode_reward,
            "avg_reward": episode_reward / max(episode_turns, 1),
        }
    
    def run_training(
        self,
        episodes: int = 10,
        turns_per_episode: int = 10,
        feedback_probability: float = 0.3,
        save_interval: int = 5,
        checkpoint_path: str = None,
    ) -> Dict:
        """
        Run full training session.
        
        Args:
            episodes: Number of episodes to run
            turns_per_episode: Turns per episode
            feedback_probability: Feedback probability
            save_interval: Save checkpoint every N episodes
            checkpoint_path: Custom checkpoint path
            
        Returns:
            Training statistics
        """
        self.start_time = time.time()
        
        print(f"\n{'=' * 50}")
        print(f"🏋️ Starting Training: {episodes} episodes × {turns_per_episode} turns")
        print(f"{'=' * 50}\n")
        
        episode_rewards = []
        
        for ep in range(1, episodes + 1):
            print(f"Episode {ep}/{episodes}")
            
            stats = self.run_episode(
                turns=turns_per_episode,
                feedback_probability=feedback_probability,
            )
            
            episode_rewards.append(stats['avg_reward'])
            
            print(f"  Total reward: {stats['reward']:.2f}, "
                  f"Avg: {stats['avg_reward']:.3f}")
            
            # Save checkpoint
            if save_interval > 0 and ep % save_interval == 0:
                path = checkpoint_path or f"checkpoints/training_ep{ep}.pt"
                self.dheera.save_checkpoint(path)
                print(f"  ✓ Checkpoint saved")
            
            print()
        
        # Final statistics
        elapsed = time.time() - self.start_time
        
        final_stats = {
            "total_episodes": self.total_episodes,
            "total_turns": self.total_turns,
            "total_reward": self.total_reward,
            "avg_reward": self.total_reward / max(self.total_turns, 1),
            "feedback_given": self.feedback_given,
            "elapsed_seconds": elapsed,
            "turns_per_second": self.total_turns / max(elapsed, 1),
            "dqn_stats": self.dheera.dqn.get_stats(),
            "rlhf_stats": self.dheera.feedback_collector.get_stats(),
        }
        
        # Print summary
        self._print_summary(final_stats)
        
        return final_stats
    
    def _print_summary(self, stats: Dict):
        """Print training summary."""
        print(f"\n{'=' * 50}")
        print("📊 Training Summary")
        print(f"{'=' * 50}")
        print(f"  Episodes: {stats['total_episodes']}")
        print(f"  Total turns: {stats['total_turns']}")
        print(f"  Total reward: {stats['total_reward']:.2f}")
        print(f"  Average reward: {stats['avg_reward']:.4f}")
        print(f"  Feedback given: {stats['feedback_given']}")
        print(f"  Time elapsed: {stats['elapsed_seconds']:.1f}s")
        print(f"  Turns/second: {stats['turns_per_second']:.1f}")
        print()
        print("DQN Stats:")
        print(f"  Training steps: {stats['dqn_stats']['total_steps']}")
        print(f"  Avg curiosity: {stats['dqn_stats']['avg_curiosity']:.4f}")
        print()
        print("RLHF Stats:")
        print(f"  Total feedback: {stats['rlhf_stats']['total_feedback']}")
        print(f"  Positive ratio: {stats['rlhf_stats']['positive_ratio']:.1%}")
        print(f"{'=' * 50}\n")
    
    def save_final_checkpoint(self, path: str = None):
        """Save final checkpoint."""
        path = path or f"checkpoints/training_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        self.dheera.save_checkpoint(path)
        print(f"✓ Final checkpoint saved: {path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dheera v0.3.0 - Training Session"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=10,
        help="Number of training episodes"
    )
    parser.add_argument(
        "--turns", "-t",
        type=int,
        default=10,
        help="Turns per episode"
    )
    parser.add_argument(
        "--feedback", "-f",
        type=float,
        default=0.3,
        help="Feedback probability (0-1)"
    )
    parser.add_argument(
        "--save-interval", "-s",
        type=int,
        default=5,
        help="Save checkpoint every N episodes (0 to disable)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode (less output)"
    )
    parser.add_argument(
        "--config", "-c",
        default="config/dheera_config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--checkpoint",
        help="Load checkpoint before training"
    )
    
    args = parser.parse_args()
    
    # Create session
    session = TrainingSession(
        config_path=args.config,
        verbose=not args.quiet,
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        session.dheera.load_checkpoint(args.checkpoint)
    
    # Run training
    session.run_training(
        episodes=args.episodes,
        turns_per_episode=args.turns,
        feedback_probability=args.feedback,
        save_interval=args.save_interval,
    )
    
    # Save final checkpoint
    session.save_final_checkpoint()


if __name__ == "__main__":
    main()
