"""
Still getting my head around all the different agents. I fully expect to refactor as I build different agents, but
I'm not sure what methods/hierarchy I'll be using yet.
"""


class AbstractAgent:
    def __init__(self, agent_info={}):
        raise NotImplementedError

    def reset(self, agent_info={}):
        """
        Setup for the agent called when the experiment first starts.
        :return: None
        """
        raise NotImplementedError

    def select_action(self, state):
        """
        Selects the next action, based on the current state.
        Public due to off-policy learning applications.
        :return:
        """
        raise NotImplementedError

    def start(self, state):
        raise NotImplementedError

    def step(self, reward, state):
        raise NotImplementedError

    def end(self, reward):
        raise NotImplementedError

    def message(self):
        pass

    def save_agent_memory(self, save_path):
        raise NotImplementedError

    def load_agent_memory(self, load_path):
        raise NotImplementedError