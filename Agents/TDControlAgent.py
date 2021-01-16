"""
Still getting my head around all the different agents. I fully expect to refactor as I build different agents, but
I'm not sure what methods/hierarchy I'll be using yet.
"""

import numpy as np
import os
import time
from enum import Enum

# Relative imports
from Encoders.RawStateEncoder import RawStateEncoder
import RLHelpers
import pickle
import os

class TdControlAlgorithm(Enum):
    QLearner = 1
    Sarsa = 2
    ExpectedSarsa = 3

class TDControlAgent:
    """
    This agent implements QLearning and SARSA.
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
        self.alpha = agent_info.get("alpha", 0.1) / self.encoder.num_tilings # step size of learning rate
        # w = state function approximation coefficients w.shape = [num_actions, feature_space]
        self.w = np.ones((self.num_actions, self.encoder.memory_allocation)) * agent_info.get("initial_weights", 0.0)
        self.algorithm = agent_info.get("algorithm", TdControlAlgorithm.QLearner)
        assert isinstance(self.algorithm, TdControlAlgorithm)
        self.name = self.algorithm.name

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
            greedy_choices = RLHelpers.all_argmax(action_values)  # act greedy
            chosen_action = np.random.choice(greedy_choices)
        # ----------------

        return chosen_action, action_values[chosen_action]

    def get_policy(self, activations):
        # Exploration
        policy = np.ones((self.num_actions))*self.epsilon/self.num_actions

        # Greedy
        action_values = np.sum(self.w[:, activations], axis=1)
        greedy_choices = RLHelpers.all_argmax(action_values) # technically
        policy[greedy_choices] += (1 - epsilon)/len(greedy_choices)

        # print("policy: ", policy)
        # print("policy.sum: ", policy.sum())
        assert .999 < policy.sum() < 1.001
        return np.array(policy)

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
        action_values = np.sum(self.w[:, activations], axis=1)
        next_action, next_action_value = self.select_action(activations) # replace optimal_next_value with actual next_value for SARSA update

        previous_action_value = np.sum(self.w[self.previous_action, self.previous_activations])

        # Update equation
        if self.algorithm == TdControlAlgorithm.QLearner:
            optimal_next_value = action_values.max() # q_learning
            delta = reward + self.gamma * optimal_next_value - previous_action_value # delta = error_term
        elif self.algorithm == TdControlAlgorithm.Sarsa:
            delta = reward + self.gamma * next_action_value - previous_action_value # delta = error_term
        elif self.algorithm == TdControlAlgorithm.ExpectedSarsa:
            delta = reward + self.gamma * np.sum(self.get_policy(activations)*next_action_value) - previous_action_value  # delta = error_term
        else:
            raise Exception("Invalid algorithm selected for TD Control Agent: %s. Select from TdControlAlgorithm enum.")

        self.w[self.previous_action][self.previous_activations] = \
            self.w[self.previous_action][self.previous_activations] + self.alpha * delta

        self.previous_action = next_action
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

    def save_agent_memory(self, save_path):
        """
        Save out any/all information the agent needs to remember to perform well. Effectively, all that it has learned.
        This generally consists of function approximators (value, action-value, policy). This does not include current
        state. It would be too annoying to sync that with the env.

        :param name: name of file to be saved.
        :param save_dir:  name of save directoy (this is populated in agent_info, and defaults to cwd if missing
        :return: NA
        """
        try:
            agent_learning = (self.encoder, self.w)
            pickle.dump(agent_learning, open( save_path, "wb" ) )
            print("Agent memory saved to: ", save_path)
        except:
            print("Unable to save agent memory to specified directory: %s " % save_path)

    def load_agent_memory(self, load_path):

        if os.path.isfile(load_path):
            agent_learning = pickle.load(open(load_path, "rb"))
            self.encoder, self.w = agent_learning
            print("Loaded agent memory from: %s" % (load_path))
            return True
        else:
            print("Warning: Unable to load agent memory. Program will proceed without loading." \
            "\n\tCould not find file: %s" % (load_path))
            return False



if __name__ == "__main__":
    from Agents import Optimizers
    import Trainer
    import gym
    from Encoders import TileCoder


    ############### Environment Setup (and configuration of agent for env) ###############
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    # ---Specific to MountainCar-v0---
    # observations = [pos, vel]
    env_name = 'MountainCar-v0'
    state_limits = np.array([[-1.2, 0.5], [-0.07, 0.07]])
    # Descent results from [16,16] to [32,32]. [64,64] gets unstable (even running 10x longer...not 100% sure why)
    feature_resolution = np.array([32,32])
    num_tilings = 32
    # Turns out some epsilon is needed, despite optimistic initialize (run with 0 until plateau, then turn it on to see!)
    epsilon = 0.01 # 0.01
    gamma = 1  # discount factor
    alpha = 1/(2**1) # learning rate: .1 to .5 Converges in a few ~1000 episodes down to about -100

    # # Specific to CartPole-v1
    # # ??? pos, vel, ang_pos, ang_vel ???
    # state_limits = np.array([[-2.5, 2.5], [-2.5, 2.5], [-0.3, 0.3], [-1, 1]])
    # feature_resolution = np.array([16,16,16,16])
    # num_grids = 32
    # epsilon = 0.01  # In case of epsilon greedy exploration
    # gamma = 1  # discount factor
    # alpha = 0.1 # learning rate

    ############### Create And Configure Agent ###############
    # Tile Coder Setup
    tile_coder = TileCoder.TileCoder(state_limits, feature_resolution, num_tilings)

    env = gym.make(env_name)

    print(os.getcwd())
    agent_info = {"encoder": tile_coder,
                  "num_actions": env.action_space.n,
                  "epsilon": epsilon,
                  "gamma": gamma,
                  "alpha": alpha,
                  "algorithm": TdControlAlgorithm.ExpectedSarsa}

    agent = TDControlAgent(agent_info)
    agent_file_path = os.getcwd() + r"/AgentMemory/Agent_%s_%s.p" % (agent.name, env_name)
    load_status = agent.load_agent_memory(agent_file_path)

    if(tile_coder.num_tilings == agent.encoder.num_tilings and
      np.all(tile_coder.tile_resolution == agent.encoder.tile_resolution) and
      np.all(tile_coder.state_boundaries == agent.encoder.state_boundaries)):
        print("Loaded encoder.")
    else:
        response = input("Unable to load agent. Previous encoding does not match. Would you like to DELETE/OVERWRITE previous agent?")
        if response.lower() == "y" or response.lower() == "yes":
            agent = QLearningAgent(agent_info)
            print("Agent data will be overwritten at next save.")
            load_status = False

    ############### Trainer Setup (load run history) ###############
    trainer_file_path = os.getcwd() + r"/../TrainingHistory/History_%s_%s.p" % (agent.name, env_name)

    trainer = Trainer.Trainer(env, agent)
    if(load_status):
        trainer.load_run_history(trainer_file_path)

    ############### Define Run inputs and Run ###############
    total_episodes = 1000
    max_steps = 1000
    render_interval = 0 # 0 is never
    trainer.train_fixed_steps(total_episodes, max_steps, render_interval) # multiple runs for up to total_steps

    ############### Save to file and plot progress ###############
    agent.save_agent_memory(agent_file_path)
    trainer.save_run_history(trainer_file_path)
    Trainer.plot(agent, np.array(trainer.rewards) )
