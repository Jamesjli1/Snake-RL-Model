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

# =========================
# Imports
# =========================
import pygame
import random
import numpy as np


# =========================
# Direction constants
# =========================
LEFT  = 0
RIGHT = 1
UP    = 2
DOWN  = 3


# =========================
# Action constants
# =========================
STRAIGHT   = 0
TURN_LEFT  = 1
TURN_RIGHT = 2


# =========================
# Snake Environment Class
# =========================
class SnakeEnv:
    """
    RL Environment containing all Snake game logic.
    This class is fully decoupled from the learning algorithm.
    """

    def __init__(self, grid_size=9, render=False):
        """
        Initialize environment parameters and rendering settings
        """
        self.grid_size = grid_size
        self.render_mode = render

        # Rendering scale
        self.window_size = 630
        self.block_size = self.window_size // self.grid_size

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


    # =========================
    # Core RL Methods
    # =========================
    def reset(self):
        """
        Reset the environment to initial state
        """
        mid = self.grid_size // 2

        self.direction = RIGHT
        self.score = 0
        self.done = False

        # Initialize snake in center
        self.snake = [
            [1 * self.block_size, mid * self.block_size],
            [2 * self.block_size, mid * self.block_size],
            [3 * self.block_size, mid * self.block_size],
        ]

        self.food = self._spawn_food()
        self.frame_iteration = 0

        return self.get_state()


    def step(self, action):
        """
        Apply an action and update environment
        """
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

        return self.get_state(), reward, self.done, self.score


    # =========================
    # Game Mechanics
    # =========================
    def _collision(self, head):
        """
        Check wall or self collision
        """
        x, y = head

        # Wall collision
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True

        # Self collision
        if head in self.snake[:-1]:
            return True

        return False


    def _spawn_food(self):
        """
        Spawn food in empty grid cell
        """
        while True:
            x = random.randrange(self.grid_size) * self.block_size
            y = random.randrange(self.grid_size) * self.block_size
            if [x, y] not in self.snake:
                return [x, y]


    def _move(self, action):
        """
        Update direction based on action and return new head position
        """
        directions = [RIGHT, DOWN, LEFT, UP]
        idx = directions.index(self.direction)

        if action == TURN_RIGHT:
            self.direction = directions[(idx + 1) % 4]
        elif action == TURN_LEFT:
            self.direction = directions[(idx - 1) % 4]

        x, y = self.snake[-1]

        if self.direction == RIGHT:
            x += self.block_size
        elif self.direction == LEFT:
            x -= self.block_size
        elif self.direction == UP:
            y -= self.block_size
        elif self.direction == DOWN:
            y += self.block_size

        return [x, y]


    # =========================
    # State Representation
    # =========================
    def get_state(self):
        """
        Convert game state into numerical representation for the agent
        """
        head = self.snake[-1]
        x, y = head

        left  = [x - self.block_size, y]
        right = [x + self.block_size, y]
        up    = [x, y - self.block_size]
        down  = [x, y + self.block_size]

        dir_left  = self.direction == LEFT
        dir_right = self.direction == RIGHT
        dir_up    = self.direction == UP
        dir_down  = self.direction == DOWN

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

        return np.array(state, dtype=int)


    # =========================
    # Rendering
    # =========================
    def render(self):
        """
        Draw the current game state using pygame
        """
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
