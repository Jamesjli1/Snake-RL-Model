"""
Author: James Li
Snake Environment for Reinforcement Learning (DQN-compatible)

This file defines the Snake game as a Reinforcement Learning environment.
It provides:
- reset(): start a new episode
- step(action): apply an action
- render(): visualize the environment

Designed to be used with Deep Q-Networks (DQN)
Last Modified: December 31, 2025
"""

"""
TESTING STATUS: NOT YET VERIFIED

Planned tests:
- Create a temporary test file that:
  - Opens the Pygame window
  - Runs one episode
  - Exits cleanly after death
- Print state vector after each step to verify correctness
- Print rewards to confirm reward structure:
  - +10 for food
  - -10 for death
  - small negative step reward
- Test edge cases:
  - Wall collision
  - Self collision
  - Food spawning not on snake
- Run multiple episodes sequentially to check reset logic
- Verify rendering correctness and performance
"""

# Imports
import pygame
import random
import numpy as np

# Direction constants
LEFT  = 0
RIGHT = 1
UP    = 2
DOWN  = 3

# Action constants
STRAIGHT   = 0
TURN_LEFT  = 1
TURN_RIGHT = 2

# Snake Environment Class
# Contains all game logic and state management
class SnakeEnv:
    # Function to initialize the environment
    def __init__(self, grid_size=9, render=False):
        
        # Rendering and grid settings
        self.grid_size = grid_size
        self.render_mode = render

        # Rendering scale
        self.window_size = 630
        self.block_size = self.window_size // self.grid_size

        # Window dimensions
        self.width = self.window_size
        self.height = self.window_size

        # Colors
        self.black = (0, 0, 0)
        self.red   = (213, 50, 80)
        self.green = (0, 255, 0)
        self.bg1   = (55, 170, 220)
        self.bg2   = (35, 140, 200)

        # Initialize pygame if rendering enabled
        if self.render_mode:
            pygame.init()
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Snake RL")
            self.clock = pygame.time.Clock()

        # Start first episode
        self.reset()

# Reset and Step are the main interaction methods
    # Function to reset the environment meaning start a new episode
    def reset(self):
        # Reset game state
        mid = self.grid_size // 2

        # Set variables to initial state
        self.direction = RIGHT
        self.score = 0
        self.done = False

        # Initialize snake in center
        self.snake = [
            [1 * self.block_size, mid * self.block_size],
            [2 * self.block_size, mid * self.block_size],
            [3 * self.block_size, mid * self.block_size],
        ]

        # Spawn initial food
        self.food = self._spawn_food()
        self.frame_iteration = 0

        return self.get_state()

    # Function to apply an action and update the environment
    def step(self, action):
        # Increment frame count
        self.frame_iteration += 1

        # Get new head position
        new_head = self._move(action)

        # Check collision
        if self._collision(new_head) or self.frame_iteration > 100 * len(self.snake):
            self.done = True
            return self.get_state(), -10, self.done, self.score

        # Insert new head
        self.snake.append(new_head)

        reward = -0.01  # step penalty

        # Check food collision
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._spawn_food()
        else:
            # Remove tail if no food eaten
            self.snake.pop(0)

        if self.render_mode:
            self.render()
        # Return state, reward, done, score
        return self.get_state(), reward, self.done, self.score


# Collision, Spawn, Movement are game mechanics methods
    # Function to check for collisions
    def _collision(self, head):
        x, y = head

        # Wall collision
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True

        # Self collision
        if head in self.snake[:-1]:
            return True

        return False

    # Function to spawn food in an empty cell
    def _spawn_food(self):
        # Randomly place food not on the snake
        while True:
            x = random.randrange(self.grid_size) * self.block_size
            y = random.randrange(self.grid_size) * self.block_size
            if [x, y] not in self.snake:
                return [x, y]

    # Function to move the snake based on action
    def _move(self, action):
        directions = [RIGHT, DOWN, LEFT, UP]
        idx = directions.index(self.direction)

        # Update direction based on action
        if action == TURN_RIGHT:
            self.direction = directions[(idx + 1) % 4]
        elif action == TURN_LEFT:
            self.direction = directions[(idx - 1) % 4]

        x, y = self.snake[-1]

        # Calculate new head position
        if self.direction == RIGHT:
            x += self.block_size
        elif self.direction == LEFT:
            x -= self.block_size
        elif self.direction == UP:
            y -= self.block_size
        elif self.direction == DOWN:
            y += self.block_size

        return [x, y]


    # Function to get the current state representation
    def get_state(self):
        # Get head position
        head = self.snake[-1]
        x, y = head

        # Define adjacent positions
        left  = [x - self.block_size, y]
        right = [x + self.block_size, y]
        up    = [x, y - self.block_size]
        down  = [x, y + self.block_size]

        # Direction flags
        dir_left  = self.direction == LEFT
        dir_right = self.direction == RIGHT
        dir_up    = self.direction == UP
        dir_down  = self.direction == DOWN

        # Feature vector of 11 binary values
        state = [
            # Danger straight
            (dir_right and self._collision(right)) or
            (dir_left  and self._collision(left)) or
            (dir_up    and self._collision(up)) or
            (dir_down  and self._collision(down)),

            # Danger right
            (dir_up    and self._collision(right)) or
            (dir_down  and self._collision(left)) or
            (dir_left  and self._collision(up)) or
            (dir_right and self._collision(down)),

            # Danger left
            (dir_down  and self._collision(right)) or
            (dir_up    and self._collision(left)) or
            (dir_right and self._collision(up)) or
            (dir_left  and self._collision(down)),

            # Current direction
            dir_left,
            dir_right,
            dir_up,
            dir_down,

            # Food location
            self.food[0] < x,
            self.food[0] > x,
            self.food[1] < y,
            self.food[1] > y,
        ]
        # Return as numpy array
        return np.array(state, dtype=int)


    # Function to render the game state using pygame
    def render(self):
        # Process pygame events
        for event in pygame.event.get():  
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.display.fill(self.black)

        # Draw checkerboard grid
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                color = self.bg1 if (row + col) % 2 == 0 else self.bg2
                pygame.draw.rect(
                    self.display,
                    color,
                    (col * self.block_size,
                     row * self.block_size,
                     self.block_size,
                     self.block_size)
                )

        # Draw food
        pygame.draw.rect(
            self.display,
            self.red,
            (*self.food, self.block_size, self.block_size)
        )

        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(
                self.display,
                self.green,
                (*segment, self.block_size, self.block_size)
            )

        pygame.display.update()
        self.clock.tick(15)
