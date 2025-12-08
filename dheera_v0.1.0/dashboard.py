#!/usr/bin/env python3
"""
Dheera Learning Dashboard
Visualize training progress and Q-values.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def load_training_stats(memory_dir: str = "./memory") -> dict:
    """Load training statistics from memory."""
    episodes_file = Path(memory_dir) / "episodes.json"
    
    if not episodes_file.exists():
        print("No training data found. Chat with Dheera first!")
        return None
    
    with open(episodes_file) as f:
        return json.load(f)


def plot_rewards(data: dict):
    """Plot reward progression over episodes."""
    episodes = data.get("episodes", {})
    
    if not episodes:
        print("No episodes to plot")
        return
    
    rewards = []
    episode_nums = []
    
    for i, (ep_id, ep_data) in enumerate(episodes.items()):
        rewards.append(ep_data.get("total_reward", 0))
        episode_nums.append(i + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Rewards over time
    plt.subplot(1, 2, 1)
    plt.plot(episode_nums, rewards, 'b-', alpha=0.5, label='Episode Reward')
    
    # Moving average
    if len(rewards) > 5:
        window = min(10, len(rewards) // 2)
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window, len(rewards) + 1), ma, 'r-', linewidth=2, label=f'Moving Avg ({window})')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Dheera Learning Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Action distribution
    plt.subplot(1, 2, 2)
    action_counts = {}
    
    for ep_data in episodes.values():
        for entry in ep_data.get("entries", []):
            action = entry.get("action_name", "UNKNOWN")
            action_counts[action] = action_counts.get(action, 0) + 1
    
    if action_counts:
        actions = list(action_counts.keys())
        counts = list(action_counts.values())
        
        # Shorten names for display
        short_names = [a[:15] for a in actions]
        
        bars = plt.barh(short_names, counts, color='steelblue')
        plt.xlabel('Times Used')
        plt.title('Action Distribution')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    str(count), va='center')
    
    plt.tight_layout()
    plt.savefig('dheera_progress.png', dpi=150)
    plt.show()
    print("\n📊 Saved to dheera_progress.png")


def print_stats(data: dict):
    """Print summary statistics."""
    episodes = data.get("episodes", {})
    
    print("\n" + "="*50)
    print("        DHEERA TRAINING STATISTICS")
    print("="*50)
    
    total_episodes = len(episodes)
    total_turns = sum(len(ep.get("entries", [])) for ep in episodes.values())
    total_reward = sum(ep.get("total_reward", 0) for ep in episodes.values())
    
    print(f"\n📈 Total Episodes: {total_episodes}")
    print(f"💬 Total Turns: {total_turns}")
    print(f"🎯 Total Reward: {total_reward:.2f}")
    
    if total_episodes > 0:
        print(f"📊 Avg Reward/Episode: {total_reward/total_episodes:.2f}")
    
    if total_turns > 0:
        print(f"📊 Avg Reward/Turn: {total_reward/total_turns:.2f}")
    
    # Action breakdown
    print("\n🎬 Action Usage:")
    action_counts = {}
    action_rewards = {}
    
    for ep_data in episodes.values():
        for entry in ep_data.get("entries", []):
            action = entry.get("action_name", "UNKNOWN")
            reward = entry.get("reward", 0)
            action_counts[action] = action_counts.get(action, 0) + 1
            action_rewards[action] = action_rewards.get(action, 0) + reward
    
    for action in sorted(action_counts.keys()):
        count = action_counts[action]
        avg_reward = action_rewards[action] / count if count > 0 else 0
        print(f"   {action[:25]:25s}: {count:4d} times, avg reward: {avg_reward:+.2f}")
    
    print("\n" + "="*50)


def main():
    print("📊 Dheera Learning Dashboard\n")
    
    data = load_training_stats()
    
    if data:
        print_stats(data)
        
        try:
            plot_rewards(data)
        except Exception as e:
            print(f"Could not create plots: {e}")
            print("(Install matplotlib: pip install matplotlib)")


if __name__ == "__main__":
    main()
