import numpy as np

class EnergyPumpingMountainCarAgent:
    """
    Obviously specific to mountain car...not a very general agent. Just available for testing/benchmark.
    """

    def __init__(self):
        self.vel = 0
        self.pos = 0
        self.last_action = 0
        self.epsilon = 0.5
        self.num_actions = 3

    def select_action(self, state):
        _, vel = state
        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(self.num_actions)  # randomly explore
        else:
            if vel >= 0:
                chosen_action = 2
            else:
                chosen_action = 0
        return chosen_action

    def start(self, state):
        self.last_action = self.select_action(self, state)
        return self.last_action

    def step(self, reward, state):
        self.last_action = self.select_action(self, state)
        return self.last_action

    def end(self, reward):
        pass

    def save_agent_memory(self, save_path):
        pass

    def load_agent_memory(self, load_path):
        return True