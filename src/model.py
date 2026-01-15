"""
Author: James Li
DQN Model for Snake Reinforcement Learning

This file defines the neural network used to approximate the Q-function in a Deep Q-Network (DQN).

The model:
- Takes a state vector as input
- Outputs Q-values for each possible action
- Is trained by the agent using experience replay

Last Modified: January 7 2026
"""

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Neural Network Model
# Approximates Q-values for given states (Q(s,a))
class LinearQNet(nn.Module):
    # Function to initialize the network
    # Dunder constructor that runs on object creation
    def __init__(self, input_size, hidden_size, output_size):
        # Call parent constructor
        super().__init__()
        # Define layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    # Function for forward pass
    def forward(self, x):

        # Apply layers with ReLU activation
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        # Return output Q-values
        return x
    
    # Function to save model weights
    def save(self, file_name="model.pth"):
        torch.save(self.state_dict(), file_name)

# Q-Trainer Class
# Handles training the Q-network using Bellman equation
class QTrainer:

    # Function to initialize the trainer
    def __init__(self, model, lr, gamma):
        # Store model, learning rate, and discount factor
        self.model = model
        self.lr = lr
        self.gamma = gamma

        # Define optimizer and loss function
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    # Function to perform one training step
    def train_step(self, state, action, reward, next_state, done):

        # Convert inputs to tensors (FAST)
        state = torch.from_numpy(np.array(state, dtype=np.float32))
        next_state = torch.from_numpy(np.array(next_state, dtype=np.float32))
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # If only one sample, add batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # Predicted Q values
        pred = self.model(state)

        # Target Q values
        target = pred.clone()

        # Update target Q values using Bellman equation
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max (self.model(next_state[idx])) # Bellman equation

            target[idx][action[idx]] = Q_new # Update target for chosen action

        # Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        # Update model weights
        loss.backward()
        self.optimizer.step()
