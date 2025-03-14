import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim, device, hidden_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.to(device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, n_actions, n_states, alpha, gamma, update_freq, hidden_dim, tn=False, er=False):
        self.n_actions = n_actions
        self.n_states = n_states
        self.gamma = gamma
        self.update_freq = update_freq

        self.tn = tn
        self.er = er
        self.memory_size = 10000
        self.batch_size = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.loss_function = nn.MSELoss().to(self.device)
        self.Q = NeuralNet(n_states, n_actions, self.device, hidden_dim)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=alpha)
        self.update_count = 0

        if tn:
            self.Q_target = NeuralNet(n_states, n_actions, self.device, hidden_dim)
            self.Q_target.load_state_dict(self.Q.state_dict())
            self.Q_target.eval()
        if er:
            # use the replay buffer to sample from every update
            self.replay_buffer = deque(maxlen=self.memory_size)
        else:
            # collect the steps up to the next update
            self.update_buffer = []

    def select_action(self, state, epsilon):  # ϵ-greedy policy
        if np.random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float, device=self.device)
            with torch.no_grad():
                q_values = self.Q.forward(state)
            action = torch.argmax(q_values).item()
        else:
            action = np.random.choice(self.n_actions)
        return action

    def update(self, state, action, reward, next_state, done):  # Q-Learning update equation
        self.update_count += 1
        if self.er:
            self.replay_buffer.append((state, action, reward, next_state, done))
            if len(self.replay_buffer) < self.batch_size:
                return
        else:
            self.update_buffer.append((state, action, reward, next_state, done))
        if self.update_count < self.update_freq:
            return

        self.update_count = 0

        if self.er:
            batch = random.sample(self.replay_buffer, self.batch_size)
            state, action, reward, next_state, done = (np.array(x) for x in zip(*batch))
        else:
            state, action, reward, next_state, done = (np.array(x) for x in zip(*self.update_buffer))
            self.update_buffer.clear()

        state = torch.tensor(state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.int, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.int, device=self.device)

        q_value = self.Q.forward(state).gather(1, action).squeeze(1)

        with torch.no_grad():
            if self.tn:
                q_next = self.Q_target.forward(next_state).max(1)[0]
            else:
                q_next = self.Q.forward(next_state).max(1)[0]

        q_target = reward + self.gamma * q_next * (1 - done)

        q_loss = self.loss_function(q_value, q_target)

        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

    def update_tn(self):
        if self.tn:
            self.Q_target.load_state_dict(self.Q.state_dict())
