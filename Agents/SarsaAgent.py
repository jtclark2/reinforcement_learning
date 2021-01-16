"""
Still getting my head around all the different agents. I fully expect to refactor as I build different agents, but
I'm not sure what methods/hierarchy I'll be using yet.
"""

import numpy as np
import os
import time

# Relative imports
from Encoders.RawStateEncoder import RawStateEncoder
import RLHelpers
import pickle


class SarsaAgent:
    """
    This agent implements Q-AgentMemory.
    """
    def __init__(self, agent_info={}):

        # Defined by the environment
        self.num_actions = agent_info["num_actions"] # number of available actions; must be defined

        # State Dependant Variables
        self.previous_activations = None
        self.previous_action = None
        self.previous_state = None

        # Encoding (this has been encapsulated in it's own class, for dependency inversion)
        self.encoder = agent_info.get("encoder", RawStateEncoder()) # no default value ( we want to fail here if encoder is not provided )

        # Hyperparameters
        self.epsilon = agent_info.get("epsilon", 0.01) # In case of epsilon greedy exploration
        self.gamma = agent_info.get("gamma", 1) # discount factor
        self.alpha = agent_info.get("alpha", 0.1) / self.encoder.num_grids # step size of learning rate
        # w = state function approximation coefficients w.shape = [num_actions, feature_space]
        self.w = np.ones((self.num_actions, self.encoder.num_activations)) * agent_info.get("initial_weights", 0.0)


    def reset(self, agent_info={}):
        """
        Setup for the agent called when the experiment first starts.
        :return: None
        """
        pass

    def select_action(self, activations):
        """
        I'd like to eventually pass this in as a policy class, but it lives here for now
        :return:
        """
        action_values = np.sum(self.w[:, activations], axis=1)

        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(self.num_actions)  # randomly explore
        else:
            chosen_action = RLHelpers.argmax(action_values)  # act greedy
        # ----------------

        return chosen_action, action_values[chosen_action]

    def start(self, state):
        activations = self.encoder.get_activations(state)
        current_action, _ = self.select_action(activations)

        self.previous_state = state
        self.previous_action = current_action
        self.previous_activations = np.copy(activations)
        return self.previous_action

    def step(self, reward, state):
        """
        Advance 1 step in the world, using SARSA update.

        :param reward: Reward value for action taken
        :param state: Current state
        :return: None
        """

        activations = self.encoder.get_activations(state)
        current_action, current_action_value = self.select_action(activations)

        previous_action_value = np.sum(self.w[self.previous_action, self.previous_activations])
        error_term = reward + self.gamma * current_action_value - previous_action_value
        self.w[self.previous_action][self.previous_activations] = \
            self.w[self.previous_action][self.previous_activations] + self.alpha * error_term

        self.previous_action = current_action
        self.previous_activations = np.copy(activations)
        return self.previous_action

    def end(self, reward):
        """
        Very last step.
        :param reward: Reward gained for action taken
        :return: None
        """
        previous_action_value = np.sum(self.w[self.previous_action, self.previous_activations])
        error_term = reward - previous_action_value
        self.w[self.previous_action][self.previous_activations] = \
            self.w[self.previous_action][self.previous_activations] + self.alpha * error_term

    def message(self):
        pass

    def save_learning(self, name=""):
        self.save_path = r"./Agents/AgentMemory/SarsaAgentLearning_%s.p" % (name) # /Agents/AgentMemory"
                         #os.path.dirname(os.path.realpath(__file__))
                         #  os.getcwd()
        print("Saving weights to: ", self.save_path)
        agent_learning = (self.encoder, self.w)
        pickle.dump(agent_learning, open( self.save_path, "wb" ) )

    def load_learning(self, name=""):
        self.load_path = r"./Agents/AgentMemory/SarsaAgentLearning_%s.p" % (name)
        if os.path.isfile(self.load_path):
            agent_learning = pickle.load(open(self.load_path, "rb"))
            print(type(agent_learning))
            self.encoder, self.w = agent_learning
            return True
        else:
            print("Warning: Unable to load file. Program will proceed without loading.")
            time.sleep(5)
            return False
