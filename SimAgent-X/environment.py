import numpy as np


class GridWorldEnv:
    """Custom Simulation Environment (Mini Gym Style)

    Grid:
    - Agent starts at (0,0)
    - Goal at (N-1,N-1)

    Actions:
    0 → Up
    1 → Down
    2 → Left
    3 → Right
    """

    def __init__(self, grid_size=5):
        self.size = grid_size
        self.state = None
        self.goal = (self.size - 1, self.size - 1)

        self.reset()

    def reset(self):
        """Reset environment and return initial observation"""
        self.state = (0, 0)
        return self.get_observation()

    def get_observation(self):
        """State → Observation Vector"""
        x, y = self.state
        return np.array([x, y], dtype=np.float32)

    def step(self, action):
        """Apply action, update state, return (obs, reward, done)"""

        x, y = self.state

        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Down
            x = min(x + 1, self.size - 1)
        elif action == 2:  # Left
            y = max(y - 1, 0)
        elif action == 3:  # Right
            y = min(y + 1, self.size - 1)

        self.state = (x, y)

        reward = self.reward_function()
        done = self.state == self.goal

        return self.get_observation(), reward, done

    def reward_function(self):
        """Reward shaping: sparse + dense"""

        if self.state == self.goal:
            return 10

        gx, gy = self.goal
        x, y = self.state

        distance = abs(gx - x) + abs(gy - y)

        return -distance * 0.1
