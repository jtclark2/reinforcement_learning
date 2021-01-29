"""
Disclaimer: I wrote this up as I 'think' it should work, but haven't actually tested. Need to look through gym to find
simple enough problem. I've been working with continuous spaces so far, not tabular.
"""

from FunctionApproximators import AbstractFunctionApproximator
import numpy as np

class TabularApproximator:

    def __init__(self, env_name, num_actions, num_states, initial_value = 0.0):
        """
        Initializes the tabular Function 'Approximator'. This actually is a perfect representation, rather than an
        approximation. This is the simplified tabular case, which just
        """
        self.num_actions = num_actions
        self.num_states = num_states
        self.env_name = env_name

        self.values = np.ones((self.num_actions, self.states)) * 0.0 # TODO: pass in initial values

    def get_values(self, state):
        """
        Although the parent method works fine, this is slightly more optimized. Gets values for all actions.
        :param state:
        :return:
        """
        action_values = np.sum(self.states[:, state], axis=1)
        return action_values

    def get_value(self, state, action):
        action_value = np.sum(self.states[action, state])
        return action_value

    def get_gradient(self, state = None, action = None):
        """
        1 for any tabular representation.
        :param state: Only included for consistency with interface.
        :param action: Only included for consistency with interface.
        :return: Gradient (which is just 1)
        """
        return 1

    def check_config_match(self, approximator):
        """
        Does this configuration match an externally provided config?

        No config required in the simple tabular case, so it matches!
        :param config:
        :return:
        """
        env_name_match = np.all(self.env_name == approximator.env_name)
        return env_name_match

    def update_weights(self, delta, state, action):
        self.states[action][state] = self.states[action][state] + delta/self._num_tilings


if __name__ == "__main__":
    print("Untested at this point.")