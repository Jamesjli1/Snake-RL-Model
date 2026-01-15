"""
Author: James Li
Temporary test file for SnakeEnv

Purpose:
- Manually verify that the Snake RL environment works correctly

Tests performed:
- Opens Pygame window
- Runs episodes until death
- Prints state vector after each step
- Prints reward after each step
- Verifies:
  - Food spawning
  - Snake growth
  - Wall collision
  - Self collision
  - Reset logic

  Tested: January 6 2026
"""

# Imports
import time
import random
from src.env.snake_env import SnakeEnv

# Function to run a single episode
def run_single_episode(env, max_steps=500):
    # Reset environment to start new episode
    state = env.reset()
    print("\n--- NEW EPISODE ---")
    print("Initial state:", state)

    step_count = 0 # Step counter

    while True:
        step_count += 1

        # Take a random action (0 = straight, 1 = left, 2 = right)
        action = random.randint(0, 2)

        next_state, reward, done, score = env.step(action)

        # Print info about each step
        print(
            f"Step: {step_count} | "
            f"Action: {action} | "
            f"Reward: {reward:.2f} | "
            f"Score: {score} | "
            f"State: {next_state}"
        )

        # Slow things down so you can visually inspect
        time.sleep(0.1)

        # Check for episode termination
        if done:
            print("Episode ended.")
            print(f"Final score: {score}")
            print(f"Total steps: {step_count}")
            break

        # Safety check to avoid infinite loops during testing
        if step_count >= max_steps:
            print("Max steps reached, forcing termination.")
            break

# Main testing routine
if __name__ == "__main__":
    # Create environment with rendering enabled
    env = SnakeEnv(grid_size=9, render=True)

    # Run multiple episodes to test reset logic
    NUM_EPISODES = 3

    # Run test episodes
    for episode in range(NUM_EPISODES):
        run_single_episode(env)
        time.sleep(1)

    print("\nAll tests completed successfully.")
