"""
Still getting my head around all the different agents. I fully expect to refactor as I build different agents, but
I'm not sure what methods/hierarchy I'll be using yet.
"""

import RLHelpers


class AbstractAgent:
    def __init__(self):
        pass

    def reset(self):
        """
        Resets state and previous step variables. May also re-load anything from __init__ that may have been corrupted.
        :return: None
        """
        raise NotImplementedError()

    def start(self, start):
        raise NotImplementedError()

        self.last_action = None
        self.previous_features = None
        return self.last_action

    def end(self):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def message(self):
        pass

    def save_state(self, path=None):
        raise NotImplementedError()

    def load_state(self, path=None):
        raise NotImplementedError()



