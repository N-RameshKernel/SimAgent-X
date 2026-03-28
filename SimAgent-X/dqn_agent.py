import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


# ✅ Q-Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


# ✅ DQN Agent
class DQNAgent:

    def __init__(self, state_dim, action_dim):

        self.state_dim = state_dim
        self.action_dim = action_dim

        # ✅ Replay Memory
        self.memory = deque(maxlen=5000)

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.lr = 0.001

        # ✅ Model
        self.model = DQN(state_dim, action_dim)

        # ✅ Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.loss_fn = nn.MSELoss()

    # ✅ Store experiences
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # ✅ Choose action
    def act(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state)

        q_values = self.model(state_tensor)

        return torch.argmax(q_values).item()

    # ✅ Replay Training Method
    def replay(self, batch_size=32):

        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:

            state_tensor = torch.FloatTensor(state)
            next_tensor = torch.FloatTensor(next_state)

            target = reward

            if not done:
                target += self.gamma * torch.max(self.model(next_tensor)).item()

            predicted = self.model(state_tensor)[action]

            target_tensor = torch.tensor(target, dtype=torch.float32)

            loss = self.loss_fn(predicted, target_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # ✅ Reduce exploration
        self.epsilon *= self.epsilon_decay
