"""
Author: James Li
Training Loop for Snake Reinforcement Learning (DQN)

This file:
- Creates the Snake environment
- Creates the DQN agent
- Runs the training loop
- Saves the best-performing model
- Plots training progress

Current Status: Working (verified with tests)
- Training loop runs successfully without crashes
- Agent score improves over time, showing learning behavior
- Mean score curve trends upward during training
- Best performing model is saved during training

Not finished:
- Loading saved models for continued training
- Save model checkpoints periodically (10, 50, 100 games) in models/ directory
    - Not needed right now
- Occasional rendering to visualize training progress
    - made later in play.py

Last Modified: January 14 2026
"""

# Imports
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import os 
from datetime import datetime

# Local imports
from src.env.snake_env import SnakeEnv
from src.agent import Agent

# Set up live plotting
plt.ion()

# For unique run IDs
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Function to plot scores
def plot(scores, mean_scores, save=False):
    plt.clf()
    plt.title("Snake DQN Training")
    plt.xlabel("Games")
    plt.ylabel("Score")

    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean Score")

    plt.legend()
    plt.ylim(ymin=0)
    plt.pause(0.001) # brief pause to update plot

    if save :
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/training_{RUN_ID}.png")

# Main training function
def train():
    # Initialize environment (no rendering during training)
    env = SnakeEnv(render=False)

    # Initialize agent
    agent = Agent()

    # Tracking variables
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    MODEL_VERSION = "v6" # Update model version as needed

    # Training loop
    while True:
        # Get initial state
        state_old = env.reset()

        done = False

        # Play one episode
        while not done:
            # Get action from agent
            action = agent.get_action(state_old)

            # Convert one-hot action to index
            move = action.index(1)

            # Perform action
            state_new, reward, done, score = env.step(move)

            # Train short memory
            agent.train_short_memory(
                state_old, move, reward, state_new, done
            )

            # Store experience
            agent.remember(
                state_old, move, reward, state_new, done
            )

            # Move to next state
            state_old = state_new

        # Episode finished
        agent.n_games += 1

        # Train long memory
        agent.train_long_memory()

        # Save model if record beaten
        if score > record:
            record = score
            agent.model.save(f"models/best_{MODEL_VERSION}.pth")

        # Logging
        scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        mean_scores.append(mean_score)

        # Terminal output
        print(
            f"Game {agent.n_games} | "
            f"Score: {score} | "
            f"Record: {record} | "
            f"Mean Score: {mean_score:.2f}"
        )

        # Plot results
        save_plot = agent.n_games % 100 == 0        # Save plot every 100 games
        plot(scores, mean_scores, save = save_plot)
        
# Main entry point
if __name__ == "__main__":
    train()
