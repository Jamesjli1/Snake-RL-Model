"""
Author: James Li
Play Script for Snake Reinforcement Learning (DQN)

This file:
- Creates the Snake environment
- Loads a trained DQN agent from a saved model (.pth)
- Runs the game with rendering so the agent can be watched in action
- Logs the score for each run

Later: 
- Render while training
    - In separate file or integrate into train.py
- Choose which saved model to load
- Screen record videos of agent performance

Last Modified: January 15 2026
"""

# Imports
import torch
import time

# Local imports
from src.env.snake_env import SnakeEnv
from src.agent import Agent

# Function to play the game using a trained agent
def play():
    # Create environment with rendering ON
    env = SnakeEnv(render=True)

    # Create agent
    agent = Agent()

    # Load trained model
    model_path = "models/best_v7.pth"
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()  # set model to evaluation mode

    print(f"Loaded model from {model_path}")

    game_count = 0

    # Loop to play multiple games
    while True:
        # Reset environment
        state = env.reset()
        done = False
        score = 0

        while not done:
            # No exploration during play
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = agent.model(state_tensor)
            move = torch.argmax(prediction).item()

            # Step environment
            state, reward, done, score = env.step(move)

            # Small delay so it’s watchable
            time.sleep(0.05)

        game_count += 1
        print(f"Game {game_count} | Score: {score}")

# Main execution
if __name__ == "__main__":
    play()

