import numpy as np

class SingleActionAgent:
    """
    Use this to test new environments. A sort of "Hello World" agent. It will always output action 0.
    """

    def __init__(self, action=0):
        self.action = action # 0 is generally inert, but

    def reset(self):
        pass

    def select_action(self, state):
        return self.action

    def start(self, state):
        return 0

    def step(self, reward, state):
        return self.select_action(state)

    def end(self, reward):
        pass

    def save(self, save_path, print_confirmation=False):
        pass

    def load(self, load_path):
        return True