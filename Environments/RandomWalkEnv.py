import random
import numpy as np

class Space:
    def __init__(self, n):
        self.n = n

class RandomWalkEnv:
    """
    Not really a complete environment (not registered, and doesn't comply with the gym interface; however,
    it supports the methods I need for this project.
    """
    def __init__(self):
        self.action_space = Space(2) # {0: left or 1: right}
        self.state = 500
        self.name = "RandomWalk"

    def reset(self):
        self.state = 500
        self.step_count = 0
        return np.array([self.state])

    def render(self):
        pass

    def step(self, action):
        self.step_count += 1

        assert action in [0,1]
        if action == 0: # left
            direction = -1
        if action == 1:
            direction = 1
        self.state += random.randint(0, 100)*direction

        # Terminate at boundaries
        done = False
        reward = 0
        if self.state <= 0:
            self.state = 0
            done = True
            reward = -1
        if self.state >= 1000:
            self.state = 1000
            done = True
            reward = 1

        info = ""

        # print(np.array([self.state]), reward, done, info)
        return np.array([self.state]), reward, done, info

    def close(self):
        pass
