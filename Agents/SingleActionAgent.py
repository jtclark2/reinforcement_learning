import numpy as np

class InertAgent:
    """
    Use this to test new environments. A sort of "Hello World" agent. It will always output action 0.
    """

    def __init__(self):
        pass

    def reset(self):
        pass

    def select_action(self, state):
        return 0

    def start(self, state):
        return 0

    def step(self, reward, state):
        return 0

    def end(self, reward):
        pass

    def save_agent_memory(self, save_path):
        pass

    def load_agent_memory(self, load_path):
        return True