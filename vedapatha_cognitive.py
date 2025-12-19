# vedapatha_cognitive.py
"""
VedaPatha Cognitive Enhancements for Dheera
Inspired by Vedic oral tradition memory systems.

Version: 0.1.0 (Prototype)
Author: Nandha Ram

Key Components:
1. DandaKramaReplayBuffer - Bidirectional experience replay
2. VikritiVerifier - Multi-head ensemble verification
3. KramaStateFeatures - Overlapping context encoding
4. VedicSpacedReplay - Leitner-style spaced repetition
"""

import random
import math
import numpy as np
from collections import deque, namedtuple, defaultdict
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
# CORE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

@dataclass
class Episode:
    """A complete trajectory of transitions."""
    transitions: List[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    episode_id: str = ""
    
    def add(self, transition: Transition):
        self.transitions.append(transition)
        self.total_reward += transition.reward
    
    def __len__(self):
        return len(self.transitions)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DAṆḌA-KRAMA REPLAY BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class DandaKramaReplayBuffer:
    """
    Bidirectional experience replay inspired by Daṇḍa-krama recitation.
    
    Daṇḍa-krama Pattern:
        For sequence A B C D:
        Recite: A B, B A, A B  |  A B C, C B A, A B C  |  A B C D, D C B A, A B C D
    
    AI Interpretation:
        - Forward pass: Learn cause → effect (state → action → reward)
        - Reverse pass: Learn effect → cause (what led to this outcome?)
        - Forward again: Reinforce the forward direction
    
    Benefits:
        - Better credit assignment
        - More robust temporal understanding
        - Stronger associative memory between states
    """
    
    def __init__(
        self,
        capacity: int = 50000,
        episode_capacity: int = 1000,
        reverse_reward_discount: float = 0.9,
    ):
        self.capacity = capacity
        self.episode_capacity = episode_capacity
        self.reverse_reward_discount = reverse_reward_discount
        
        # Standard transition buffer
        self.buffer: deque = deque(maxlen=capacity)
        
        # Episode storage for trajectory-based sampling
        self.episodes: deque = deque(maxlen=episode_capacity)
        self.current_episode: Optional[Episode] = None
        
        # Statistics
        self.danda_samples = 0
        self.forward_samples = 0
        self.reverse_samples = 0
    
    def start_episode(self, episode_id: str = ""):
        """Begin a new episode."""
        self.current_episode = Episode(episode_id=episode_id)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition to buffer and current episode."""
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)
        
        if self.current_episode is not None:
            self.current_episode.add(transition)
    
    def end_episode(self):
        """Complete current episode and store it."""
        if self.current_episode and len(self.current_episode) > 0:
            self.episodes.append(self.current_episode)
        self.current_episode = None
    
    def sample_standard(self, batch_size: int) -> List[Transition]:
        """Standard uniform sampling."""
        self.forward_samples += batch_size
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def sample_danda(self, batch_size: int) -> List[Transition]:
        """
        Daṇḍa-krama style sampling: Forward → Reverse → Forward pattern.
        
        Returns interleaved transitions from forward and reversed trajectories.
        """
        if len(self.episodes) == 0:
            return self.sample_standard(batch_size)
        
        self.danda_samples += batch_size
        samples = []
        
        # Sample from episodes
        while len(samples) < batch_size:
            # Pick a random episode
            episode = random.choice(self.episodes)
            
            if len(episode) < 2:
                continue
            
            # Get a contiguous subsequence
            start = random.randint(0, max(0, len(episode) - 3))
            end = min(start + 5, len(episode))  # Max 5 transitions
            
            forward = episode.transitions[start:end]
            reverse = self._create_reverse_trajectory(forward)
            
            # Daṇḍa pattern: F, R, F for each position
            for f, r in zip(forward, reverse):
                samples.append(f)  # Forward
                samples.append(r)  # Reverse
                samples.append(f)  # Forward again (reinforcement)
                
                if len(samples) >= batch_size:
                    break
        
        return samples[:batch_size]
    
    def _create_reverse_trajectory(
        self, 
        forward: List[Transition]
    ) -> List[Transition]:
        """
        Create time-reversed trajectory.
        
        In the reverse trajectory:
        - state and next_state are swapped
        - rewards are discounted (reverse causality is weaker)
        - actions are kept (no inverse action for now)
        """
        self.reverse_samples += len(forward)
        
        reverse = []
        for t in reversed(forward):
            reverse.append(Transition(
                state=t.next_state,
                action=t.action,  # Could use inverse action mapping
                reward=t.reward * self.reverse_reward_discount,
                next_state=t.state,
                done=False  # Reverse trajectory is not "done"
            ))
        return reverse
    
    def sample_mixed(self, batch_size: int, danda_ratio: float = 0.3) -> List[Transition]:
        """Mix of standard and Daṇḍa sampling."""
        danda_size = int(batch_size * danda_ratio)
        standard_size = batch_size - danda_size
        
        samples = []
        if danda_size > 0 and len(self.episodes) > 0:
            samples.extend(self.sample_danda(danda_size))
        if standard_size > 0:
            samples.extend(self.sample_standard(standard_size))
        
        random.shuffle(samples)
        return samples
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "buffer_size": len(self.buffer),
            "episode_count": len(self.episodes),
            "danda_samples": self.danda_samples,
            "forward_samples": self.forward_samples,
            "reverse_samples": self.reverse_samples,
            "danda_ratio": self.danda_samples / max(1, self.danda_samples + self.forward_samples),
        }
    
    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VIKRITI VERIFIER (Multi-Head Ensemble)
# ═══════════════════════════════════════════════════════════════════════════════

class TinyQHead(nn.Module):
    """Single Q-network head."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VikritiVerifier(nn.Module):
    """
    Multi-head ensemble verification inspired by Vikriti pathas.
    
    Vikriti Concept:
        The 8 Vikriti pathas (Jaṭā, Mālā, Śikhā, Rekhā, Dhvaja, Daṇḍa, Rath, Ghana)
        provide 8 different "paths" to verify the same knowledge.
        
        If all paths agree → high confidence
        If paths diverge → uncertainty, need re-examination
    
    AI Interpretation:
        Multiple Q-network heads with different initializations.
        Agreement = confidence, disagreement = exploration signal.
    
    Benefits:
        - Principled uncertainty estimation
        - Better exploration (explore when uncertain)
        - More robust action selection
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_heads: int = 4,  # Like 4 major Vikriti pathas
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_heads = num_heads
        
        # Create ensemble of Q-heads
        self.q_heads = nn.ModuleList([
            TinyQHead(state_dim, action_dim, hidden_dim)
            for _ in range(num_heads)
        ])
        
        # Statistics
        self.agreement_history = deque(maxlen=1000)
        self.confidence_history = deque(maxlen=1000)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Get mean Q-values across all heads."""
        q_values = torch.stack([head(state) for head in self.q_heads])
        return q_values.mean(dim=0)
    
    def forward_with_uncertainty(
        self, 
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values with uncertainty estimates."""
        q_values = torch.stack([head(state) for head in self.q_heads])
        
        q_mean = q_values.mean(dim=0)
        q_std = q_values.std(dim=0)
        
        return q_mean, q_std
    
    def select_action(
        self,
        state: torch.Tensor,
        epsilon: float = 0.0,
        use_uncertainty: bool = True,
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        Select action with optional uncertainty-aware exploration.
        
        Returns:
            action: Selected action
            confidence: Confidence in selection (0-1)
            info: Additional information
        """
        with torch.no_grad():
            q_mean, q_std = self.forward_with_uncertainty(state)
            
            # Squeeze batch dimension if present
            q_mean = q_mean.squeeze(0)  # (action_dim,)
            q_std = q_std.squeeze(0)    # (action_dim,)
            
            # Best action by mean Q-value
            best_action = q_mean.argmax().item()
            
            # Confidence = inverse of relative uncertainty
            # High std relative to Q-value magnitude = low confidence
            q_magnitude = q_mean.abs().mean().item() + 1e-6
            relative_std = q_std[best_action].item() / q_magnitude
            confidence = 1.0 / (1.0 + relative_std)
            
            # Check agreement: do all heads agree on best action?
            head_actions = [head(state).argmax(dim=-1).item() for head in self.q_heads]
            agreement = sum(a == best_action for a in head_actions) / self.num_heads
            
            # Record stats
            self.agreement_history.append(agreement)
            self.confidence_history.append(confidence)
            
            # Exploration decision
            if random.random() < epsilon:
                # Standard epsilon exploration
                action = random.randint(0, self.action_dim - 1)
                confidence = 0.0
            elif use_uncertainty and confidence < 0.5 and agreement < 0.75:
                # Uncertainty-driven exploration: pick action with highest uncertainty
                action = q_std.argmax().item()
                confidence = 0.3  # Mark as exploratory
            else:
                action = best_action
        
        info = {
            "q_mean": q_mean.cpu().numpy(),
            "q_std": q_std.cpu().numpy(),
            "agreement": agreement,
            "head_actions": head_actions,
            "exploratory": confidence < 0.5,
        }
        
        return action, confidence, info
    
    def get_training_targets(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
    ) -> List[torch.Tensor]:
        """
        Get training targets for each head.
        
        Uses different target strategies for diversity:
        - Half use their own next Q-values (independent)
        - Half use ensemble mean (cooperative)
        """
        targets_list = []
        
        with torch.no_grad():
            # Ensemble mean for next state
            next_q_mean = self.forward(next_states)
            next_q_max_mean = next_q_mean.max(dim=-1)[0]
        
        for i, head in enumerate(self.q_heads):
            with torch.no_grad():
                next_q = head(next_states)
                next_q_max_own = next_q.max(dim=-1)[0]
            
            # Alternate between own and mean targets
            if i % 2 == 0:
                next_q_max = next_q_max_own
            else:
                next_q_max = next_q_max_mean
            
            target = rewards + gamma * next_q_max * (1 - dones)
            targets_list.append(target)
        
        return targets_list
    
    def get_stats(self) -> Dict[str, float]:
        """Get verification statistics."""
        return {
            "mean_agreement": np.mean(self.agreement_history) if self.agreement_history else 0.0,
            "mean_confidence": np.mean(self.confidence_history) if self.confidence_history else 0.0,
            "std_agreement": np.std(self.agreement_history) if self.agreement_history else 0.0,
            "std_confidence": np.std(self.confidence_history) if self.confidence_history else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. KRAMA STATE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

class KramaStateFeatures:
    """
    Overlapping context features inspired by Krama patha.
    
    Krama Pattern:
        Words: A B C D E
        Krama: A-B | B-C | C-D | D-E
        
        Each element appears in TWO pairs → redundancy + context linking.
    
    AI Interpretation:
        Create features that capture TRANSITIONS between states/messages,
        not just the current state. This captures conversation FLOW.
    
    Benefits:
        - Detects topic shifts, sentiment changes
        - Better context for action selection
        - Captures temporal patterns
    """
    
    def __init__(self, feature_dim: int = 8):
        self.feature_dim = feature_dim
        self.history: deque = deque(maxlen=10)
    
    def add_state(self, state: np.ndarray, message: str = ""):
        """Add a state to history."""
        self.history.append({
            "state": state,
            "message": message,
        })
    
    def compute_krama_features(self) -> np.ndarray:
        """
        Compute Krama-style overlapping features from state history.
        
        Returns:
            np.ndarray of shape (feature_dim,)
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        if len(self.history) < 2:
            return features
        
        # Get recent state pairs
        pairs = []
        for i in range(len(self.history) - 1):
            pairs.append((
                self.history[i]["state"],
                self.history[i + 1]["state"]
            ))
        
        # Feature 0-1: Mean state delta (how much is state changing?)
        deltas = [s2 - s1 for s1, s2 in pairs]
        mean_delta = np.mean([np.abs(d).mean() for d in deltas])
        features[0] = np.tanh(mean_delta)  # Normalize to [-1, 1]
        
        # Feature 2-3: Delta trend (is change accelerating?)
        if len(deltas) >= 2:
            delta_norms = [np.linalg.norm(d) for d in deltas]
            delta_diff = delta_norms[-1] - delta_norms[0]
            features[1] = np.tanh(delta_diff)
        
        # Feature 4-5: State oscillation (back-and-forth patterns)
        if len(pairs) >= 2:
            directions = [np.sign(d) for d in deltas]
            oscillation = sum(
                (directions[i] * directions[i+1]).sum() < 0
                for i in range(len(directions) - 1)
            ) / max(1, len(directions) - 1)
            features[2] = oscillation
        
        # Feature 6-7: Specific dimension tracking (e.g., sentiment, complexity)
        if len(self.history) >= 2:
            # Track sentiment dimension (assuming index 1 in state)
            sent_history = [h["state"][1] for h in self.history]
            features[3] = np.mean(sent_history)  # Mean sentiment
            features[4] = np.std(sent_history)   # Sentiment stability
            
            # Track complexity dimension (assuming index 2)
            comp_history = [h["state"][2] for h in self.history]
            features[5] = np.mean(comp_history)
            features[6] = comp_history[-1] - comp_history[0]  # Complexity trend
        
        # Feature 7: Conversation momentum
        if len(self.history) >= 3:
            recent_deltas = [np.linalg.norm(d) for d in deltas[-3:]]
            momentum = sum(recent_deltas) / len(recent_deltas)
            features[7] = np.tanh(momentum)
        
        return features
    
    def augment_state(self, current_state: np.ndarray) -> np.ndarray:
        """Augment current state with Krama features."""
        krama_features = self.compute_krama_features()
        return np.concatenate([current_state, krama_features])
    
    def reset(self):
        """Clear history."""
        self.history.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VEDIC SPACED REPLAY (Leitner System)
# ═══════════════════════════════════════════════════════════════════════════════

class VedicSpacedReplay:
    """
    Spaced repetition replay inspired by Vedic ritualized training.
    
    Vedic Practice:
        Not random repetition — structured cycles:
        - Daily morning recitation
        - Evening review
        - Weekly full recitation
        - Periodic deep practice sessions
    
    AI Interpretation:
        Leitner-style box system for experience replay:
        - Box 0: New/difficult experiences (sample often)
        - Box 1-2: Learning experiences (sample medium)
        - Box 3-4: Mastered experiences (sample rarely)
        
        Experiences move between boxes based on TD-error.
    
    Benefits:
        - Focus compute on what needs learning
        - Don't over-train on mastered experiences
        - Natural curriculum emerges
    """
    
    def __init__(
        self,
        num_boxes: int = 5,
        box_capacities: Optional[List[int]] = None,
    ):
        self.num_boxes = num_boxes
        
        # Default capacities: larger for newer boxes
        if box_capacities is None:
            box_capacities = [10000, 5000, 2000, 1000, 500]
        
        self.boxes = [deque(maxlen=cap) for cap in box_capacities[:num_boxes]]
        
        # Sampling probabilities: higher for earlier boxes
        self.sample_probs = self._compute_sample_probs()
        
        # TD-error tracking
        self.td_errors: Dict[int, float] = {}  # transition_id → td_error
        self.next_id = 0
        
        # Statistics
        self.promotions = 0
        self.demotions = 0
    
    def _compute_sample_probs(self) -> np.ndarray:
        """Compute sampling probabilities (exponential decay)."""
        probs = np.array([2 ** (self.num_boxes - 1 - i) for i in range(self.num_boxes)])
        return probs / probs.sum()
    
    def add(self, transition: Transition, initial_td_error: float = 1.0):
        """Add new transition to box 0."""
        trans_id = self.next_id
        self.next_id += 1
        
        self.boxes[0].append((trans_id, transition))
        self.td_errors[trans_id] = initial_td_error
    
    def sample(self, batch_size: int) -> List[Tuple[int, Transition]]:
        """Sample with spaced repetition probabilities."""
        samples = []
        
        # Adjust probs based on box sizes
        box_sizes = [len(box) for box in self.boxes]
        available_probs = np.array([
            self.sample_probs[i] if box_sizes[i] > 0 else 0
            for i in range(self.num_boxes)
        ])
        
        if available_probs.sum() == 0:
            return []
        
        available_probs /= available_probs.sum()
        
        for _ in range(batch_size):
            # Select box
            box_idx = np.random.choice(self.num_boxes, p=available_probs)
            
            if len(self.boxes[box_idx]) > 0:
                # Sample from box
                trans_id, transition = random.choice(self.boxes[box_idx])
                samples.append((trans_id, transition))
        
        return samples
    
    def update_td_error(self, trans_id: int, new_td_error: float):
        """
        Update TD-error and potentially move transition between boxes.
        
        Logic:
        - If TD-error decreased significantly → promote to higher box (less frequent)
        - If TD-error increased → demote to lower box (more frequent)
        """
        if trans_id not in self.td_errors:
            return
        
        old_error = self.td_errors[trans_id]
        self.td_errors[trans_id] = new_td_error
        
        # Find current box
        current_box = None
        for box_idx, box in enumerate(self.boxes):
            for i, (tid, trans) in enumerate(box):
                if tid == trans_id:
                    current_box = box_idx
                    break
            if current_box is not None:
                break
        
        if current_box is None:
            return
        
        # Decide on promotion/demotion
        if new_td_error < old_error * 0.5:
            # Learned well → promote
            new_box = min(current_box + 1, self.num_boxes - 1)
            self.promotions += 1
        elif new_td_error > old_error * 1.5:
            # Forgotten → demote
            new_box = max(current_box - 1, 0)
            self.demotions += 1
        else:
            new_box = current_box
        
        # Move if needed
        if new_box != current_box:
            self._move_transition(trans_id, current_box, new_box)
    
    def _move_transition(self, trans_id: int, from_box: int, to_box: int):
        """Move transition between boxes."""
        # Find and remove from current box
        transition = None
        for i, (tid, trans) in enumerate(self.boxes[from_box]):
            if tid == trans_id:
                transition = trans
                del self.boxes[from_box][i]
                break
        
        # Add to new box
        if transition is not None:
            self.boxes[to_box].append((trans_id, transition))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get replay statistics."""
        box_sizes = [len(box) for box in self.boxes]
        total = sum(box_sizes)
        
        return {
            "box_sizes": box_sizes,
            "total_transitions": total,
            "box_percentages": [s / max(1, total) * 100 for s in box_sizes],
            "promotions": self.promotions,
            "demotions": self.demotions,
            "promotion_rate": self.promotions / max(1, self.promotions + self.demotions),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. INTEGRATED VEDAPATHA MEMORY SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class VedaPathaMemory:
    """
    Unified memory system combining all Vedic-inspired components.
    
    Components:
    1. DandaKrama replay for bidirectional learning
    2. Vikriti verification for uncertainty-aware action selection
    3. Krama features for context tracking
    4. Spaced repetition for efficient learning
    
    This is the "cognitive layer" that sits between raw experience
    and the DQN agent, providing richer representations and smarter replay.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict[str, Any]] = None,
    ):
        config = config or {}
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize components
        self.danda_buffer = DandaKramaReplayBuffer(
            capacity=config.get("buffer_capacity", 50000),
            episode_capacity=config.get("episode_capacity", 1000),
        )
        
        self.vikriti = VikritiVerifier(
            state_dim=state_dim + config.get("krama_dim", 8),  # Augmented state
            action_dim=action_dim,
            num_heads=config.get("num_heads", 4),
        )
        
        self.krama_features = KramaStateFeatures(
            feature_dim=config.get("krama_dim", 8)
        )
        
        self.spaced_replay = VedicSpacedReplay(
            num_boxes=config.get("num_boxes", 5)
        )
        
        # Mode flags
        self.use_danda = config.get("use_danda", True)
        self.use_vikriti = config.get("use_vikriti", True)
        self.use_krama = config.get("use_krama", True)
        self.use_spaced = config.get("use_spaced", True)
    
    def process_state(self, state: np.ndarray, message: str = "") -> np.ndarray:
        """Process state through Krama feature augmentation."""
        if self.use_krama:
            self.krama_features.add_state(state, message)
            return self.krama_features.augment_state(state)
        return state
    
    def select_action(
        self,
        state: np.ndarray,
        epsilon: float = 0.0,
    ) -> Tuple[int, float, Dict[str, Any]]:
        """Select action using Vikriti verification."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        if self.use_vikriti:
            return self.vikriti.select_action(state_tensor, epsilon)
        else:
            q_values = self.vikriti(state_tensor)
            action = q_values.argmax().item()
            return action, 1.0, {}
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: float = 1.0,
    ):
        """Store transition in both replay systems."""
        transition = Transition(state, action, reward, next_state, done)
        
        # DandaKrama buffer
        self.danda_buffer.push(state, action, reward, next_state, done)
        
        # Spaced replay
        if self.use_spaced:
            self.spaced_replay.add(transition, td_error)
    
    def sample_batch(
        self,
        batch_size: int,
        mode: str = "mixed"
    ) -> List[Transition]:
        """
        Sample a batch of transitions.
        
        Modes:
        - "standard": Uniform sampling
        - "danda": Bidirectional DandaKrama sampling
        - "spaced": Spaced repetition sampling
        - "mixed": Combination of all
        """
        if mode == "standard":
            return self.danda_buffer.sample_standard(batch_size)
        elif mode == "danda":
            return self.danda_buffer.sample_danda(batch_size)
        elif mode == "spaced":
            samples = self.spaced_replay.sample(batch_size)
            return [trans for _, trans in samples]
        else:  # mixed
            # 40% danda, 30% spaced, 30% standard
            danda_size = int(batch_size * 0.4)
            spaced_size = int(batch_size * 0.3)
            standard_size = batch_size - danda_size - spaced_size
            
            samples = []
            if danda_size > 0:
                samples.extend(self.danda_buffer.sample_danda(danda_size))
            if spaced_size > 0:
                spaced = self.spaced_replay.sample(spaced_size)
                samples.extend([trans for _, trans in spaced])
            if standard_size > 0:
                samples.extend(self.danda_buffer.sample_standard(standard_size))
            
            random.shuffle(samples)
            return samples
    
    def start_episode(self, episode_id: str = ""):
        """Start a new episode."""
        self.danda_buffer.start_episode(episode_id)
        self.krama_features.reset()
    
    def end_episode(self):
        """End current episode."""
        self.danda_buffer.end_episode()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get stats from all components."""
        return {
            "danda_buffer": self.danda_buffer.get_stats(),
            "vikriti": self.vikriti.get_stats() if self.use_vikriti else {},
            "spaced_replay": self.spaced_replay.get_stats() if self.use_spaced else {},
        }


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("🕉️  VedaPatha Cognitive Enhancements - Test Suite")
    print("=" * 70)
    
    # Test parameters
    state_dim = 16
    action_dim = 7
    
    # 1. Test DandaKrama Buffer
    print("\n📿 Testing DandaKrama Replay Buffer...")
    danda = DandaKramaReplayBuffer()
    
    # Create some episodes
    for ep in range(5):
        danda.start_episode(f"episode_{ep}")
        for step in range(10):
            state = np.random.randn(state_dim).astype(np.float32)
            action = np.random.randint(action_dim)
            reward = np.random.random()
            next_state = np.random.randn(state_dim).astype(np.float32)
            danda.push(state, action, reward, next_state, done=(step == 9))
        danda.end_episode()
    
    samples = danda.sample_danda(32)
    print(f"   ✓ Sampled {len(samples)} transitions with DandaKrama pattern")
    print(f"   Stats: {danda.get_stats()}")
    
    # 2. Test Vikriti Verifier
    print("\n🔱 Testing Vikriti Verifier...")
    vikriti = VikritiVerifier(state_dim, action_dim, num_heads=4)
    
    test_state = torch.randn(1, state_dim)
    action, confidence, info = vikriti.select_action(test_state)
    print(f"   ✓ Selected action {action} with confidence {confidence:.3f}")
    print(f"   Agreement across heads: {info['agreement']:.2f}")
    print(f"   Head actions: {info['head_actions']}")
    
    # 3. Test Krama Features
    print("\n🌊 Testing Krama State Features...")
    krama = KramaStateFeatures()
    
    for i in range(5):
        state = np.random.randn(state_dim).astype(np.float32)
        krama.add_state(state, f"message_{i}")
    
    features = krama.compute_krama_features()
    print(f"   ✓ Computed {len(features)} Krama features")
    print(f"   Features: {features.round(3)}")
    
    # 4. Test Spaced Replay
    print("\n📚 Testing Vedic Spaced Replay...")
    spaced = VedicSpacedReplay()
    
    for i in range(100):
        trans = Transition(
            np.random.randn(state_dim).astype(np.float32),
            np.random.randint(action_dim),
            np.random.random(),
            np.random.randn(state_dim).astype(np.float32),
            False
        )
        spaced.add(trans, initial_td_error=np.random.random())
    
    samples = spaced.sample(32)
    print(f"   ✓ Sampled {len(samples)} transitions")
    print(f"   Stats: {spaced.get_stats()}")
    
    # 5. Test Integrated System
    print("\n🏛️  Testing Integrated VedaPatha Memory...")
    vedapatha = VedaPathaMemory(state_dim, action_dim)
    
    vedapatha.start_episode("test")
    for i in range(20):
        state = np.random.randn(state_dim).astype(np.float32)
        augmented = vedapatha.process_state(state, f"turn_{i}")
        action, conf, info = vedapatha.select_action(augmented)
        next_state = np.random.randn(state_dim).astype(np.float32)
        vedapatha.store_transition(augmented, action, np.random.random(), 
                                   vedapatha.process_state(next_state), 
                                   done=(i == 19))
    vedapatha.end_episode()
    
    batch = vedapatha.sample_batch(32, mode="mixed")
    print(f"   ✓ Sampled {len(batch)} transitions with mixed strategy")
    print(f"   Comprehensive stats: {vedapatha.get_comprehensive_stats()}")
    
    print("\n" + "=" * 70)
    print("✅ All VedaPatha components tested successfully!")
    print("=" * 70)
