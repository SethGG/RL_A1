import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

        self.to(device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, n_actions, n_states, epsilon, alpha, gamma):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_function = nn.MSELoss().to(self.device)
        self.Q = NeuralNet(n_states, n_actions, self.device)
        self.optimizer = optim.SGD(self.Q.parameters(), lr=alpha)

    def select_action(self, state):  # ϵ-greedy policy
        if np.random.random() > self.epsilon:
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.Q.forward(state)
            action = torch.argmax(q_values).item()
        else:
            action = np.random.choice(self.n_actions)
        return action

    def update(self, state, action, reward, next_state):  # Q-Learning update equation
        state = torch.FloatTensor(state).to(self.device)
        action = torch.Tensor(action).to(self.device)
        reward = torch.Tensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)

        q_value = self.Q.forward(state)[action]
        q_next = self.Q.forward(next_state).max()

        td_target = reward + self.gamma * q_next
        td_delta = self.loss_function(q_value, td_target)

        self.optimizer.zero_grad()
        td_delta.backward()
        self.optimizer.step()
