"""
Author: James Li
DQN Agent for Snake Reinforcement Learning

The Agent:
- Chooses actions (exploration vs exploitation)
- Stores experiences
- Trains the Q-network using replay memory

The agent connects:
Environment to Model to Trainer

Last Modified: January 9 2026
"""

# Imports
import random
import torch
import numpy as np
from collections import deque # for fast memory storage

# Local imports from model.py
from src.model import LinearQNet, QTrainer

# Agent Class that interacts with the environment and learns
class Agent:
    # Function to initialize the agent
    def __init__(self):
        # Initialize variables
        self.n_games = 0        # number of games played

        # Exploration parameters for bellman equation
        self.epsilon = 0        # controls randomness
        self.gamma = 0.9        # how much to discount future rewards

        # Replay memory
        self.memory = deque(maxlen=100_000)

        # Model 
        self.model = LinearQNet(
            input_size=11,     # state size from environment
            hidden_size=256,
            output_size=3      # [straight, left, right]
        )

        # Trainer 
        self.trainer = QTrainer(
            self.model,
            lr=0.001,           # learning rate
            gamma=self.gamma
        )

    # Function to store experience in memory
    def remember(self, state, action, reward, next_state, done):
        # Store experience tuple
        self.memory.append((state, action, reward, next_state, done))

    # Function to train the model using replay memory
    def train_long_memory(self):
        # Initialize batch size
        BATCH_SIZE = 1000

        # Sample a batch from memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        # Unzip the batch into separate components
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        # Train the model on the batch
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # Function to train on the most recent experience
    def train_short_memory(self, state, action, reward, next_state, done):
        # Train on single experience
        self.trainer.train_step(state, action, reward, next_state, done)

    # Function to decide action based on current state
    def get_action(self, state):
     
        # Exploration decreases as games increase
        self.epsilon = max(5, 200 - self.n_games)

        final_move = [0, 0, 0] # [straight, left, right]

        # Random move (exploration)
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Predict Q-values (exploitation)
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1 

        return final_move
