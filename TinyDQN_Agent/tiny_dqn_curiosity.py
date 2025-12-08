"""
tiny_dqn_curiosity.py

Minimal Tiny DQN agent with:
- Extrinsic reward ("dopamine" from env)
- Intrinsic curiosity reward via Random Network Distillation (RND)

You can plug any env with a Gym-like interface:
    obs = env.reset()
    obs, reward, done, info = env.step(action)
"""

import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ===============================
# 1. Simple MLP Network for DQN
# ===============================

class TinyDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ========================================
# 2. Random Network Distillation (Curiosity)
# ========================================

class RNDModule(nn.Module):
    """
    Random Network Distillation:
    - target: fixed random network (no training)
    - predictor: trainable network trying to match target
    - intrinsic reward = prediction error (novelty)
    """
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        # Fixed random target network
        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Predictor network
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Freeze target params
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, state_batch):
        """
        state_batch: (B, state_dim) tensor
        returns: intrinsic reward per state (B,) tensor
        """
        with torch.no_grad():
            target_feat = self.target(state_batch)
        pred_feat = self.predictor(state_batch)
        # Mean squared error along feature dimension
        mse = (target_feat - pred_feat).pow(2).mean(dim=1)
        return mse

    def rnd_loss(self, state_batch):
        """
        Loss used to train predictor network.
        """
        with torch.no_grad():
            target_feat = self.target(state_batch)
        pred_feat = self.predictor(state_batch)
        return (target_feat - pred_feat).pow(2).mean()


# ==========================
# 3. Replay Buffer
# ==========================

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done')
)

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


# ==========================
# 4. DQN Agent with Curiosity
# ==========================

class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_capacity=100_000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=50_000,
        target_update_freq=1000,
        curiosity_coef=0.1,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_capacity)
        self.curiosity_coef = curiosity_coef

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # DQN networks
        self.policy_net = TinyDQN(state_dim, action_dim).to(self.device)
        self.target_net = TinyDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Curiosity RND module & its optimizer
        self.rnd = RNDModule(state_dim).to(self.device)
        self.rnd_optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=lr)

        # Epsilon-greedy scheduling
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

        self.target_update_freq = target_update_freq

    def select_action(self, state_np):
        """
        state_np: np.array of shape (state_dim,)
        """
        self.total_steps += 1
        # Epsilon decay
        t = min(self.total_steps, self.epsilon_decay_steps)
        frac = t / self.epsilon_decay_steps
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward_ext, next_state, done):
        """
        - Compute intrinsic reward from next_state (novelty).
        - Combine extrinsic + curiosity reward into total reward.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Curiosity reward from next_state (can also use state)
            intrinsic = self.rnd(next_state_t).item()

        reward_total = reward_ext + self.curiosity_coef * intrinsic

        self.buffer.push(state, action, reward_total, next_state, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None  # not enough data yet

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*transitions)

        # Convert to tensors
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # 1. DQN loss
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q = reward_batch + self.gamma * next_q_values * (1.0 - done_batch)

        dqn_loss = (q_values - target_q).pow(2).mean()

        self.optimizer.zero_grad()
        dqn_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # 2. RND curiosity loss (train predictor)
        rnd_loss = self.rnd.rnd_loss(next_state_batch)
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        nn.utils.clip_grad_norm_(self.rnd.predictor.parameters(), 1.0)
        self.rnd_optimizer.step()

        # 3. Target network sync
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return {
            "dqn_loss": dqn_loss.item(),
            "rnd_loss": rnd_loss.item(),
            "epsilon": self.epsilon,
        }


# ==========================================
# 5. Simple Toy Environment (Replace Later)
# ==========================================

class SimpleLineEnv:
    """
    Tiny toy env just so you can run the script.

    State: 1D position x in [-10, 10]
    Actions: 0 = -1 step, 1 = +1 step
    Goal: reach x = +10
    Reward:
        +1 when moving closer to 10,
        +10 when exactly at 10,
        -1 when moving away.
    Episode ends when |x| > 12 or step limit reached.
    """

    def __init__(self, max_steps=50):
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.x = random.randint(-5, 5)
        self.steps = 0
        return np.array([self.x], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        if action == 0:
            self.x -= 1
        else:
            self.x += 1

        # Reward based on moving toward +10
        prev_dist = abs((self.x + (1 if action == 0 else -1)) - 10)
        new_dist = abs(self.x - 10)

        reward = 0.0
        if self.x == 10:
            reward = 10.0
        elif new_dist < prev_dist:
            reward = 1.0
        else:
            reward = -1.0

        done = False
        if self.steps >= self.max_steps or abs(self.x) > 12:
            done = True

        state = np.array([self.x], dtype=np.float32)
        return state, reward, done, {}



# ==========================
# 6. Training Loop Example
# ==========================

def train_tiny_dqn():
    env = SimpleLineEnv()
    state_dim = 1
    action_dim = 2

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_capacity=50_000,
        curiosity_coef=0.05,  # tune this if needed
    )

    num_episodes = 1000

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        ep_reward = 0.0

        while True:
            action = agent.select_action(state)
            next_state, reward_ext, done, _ = env.step(action)
            agent.store_transition(state, action, reward_ext, next_state, float(done))
            train_info = agent.train_step()

            state = next_state
            ep_reward += reward_ext

            if done:
                break

        if ep % 20 == 0:
            if train_info is not None:
                print(
                    f"Ep {ep:4d} | Reward: {ep_reward:6.2f} | "
                    f"DQN Loss: {train_info['dqn_loss']:.4f} | "
                    f"RND Loss: {train_info['rnd_loss']:.4f} | "
                    f"Eps: {train_info['epsilon']:.3f}"
                )
            else:
                print(f"Ep {ep:4d} | Reward: {ep_reward:6.2f} | Warming up...")


if __name__ == "__main__":
    train_tiny_dqn()

