"""
Still getting my head around all the different agents. I fully expect to refactor as I build different agents, but
I'm not sure what methods/hierarchy I'll be using yet.
"""

import numpy as np
from enum import Enum

# Relative imports
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
        self.previous_action = None
        self.previous_state = None

        # Encoding (this has been encapsulated in it's own class, for dependency inversion)
        self.value_approximator = agent_info["state_action_value_approximator"]

        # Hyperparameters
        self.epsilon = agent_info.get("epsilon", 0.01) # In case of epsilon greedy exploration
        self.gamma = agent_info.get("gamma", 1) # discount factor
        self.alpha = agent_info["alpha"] # step size of learning rate

        self.off_policy_agent = agent_info.get("off_policy_agent", None)
        self.algorithm = agent_info.get("algorithm", TdControlAlgorithm.QLearner)
        assert isinstance(self.algorithm, TdControlAlgorithm)
        self.name = self.algorithm.name

    def reset(self, agent_info={}):
        """
        Setup for the agent called when the experiment first starts.
        :return: None
        """
        pass

    def select_action(self, state):
        """
        Selects the next action, based on the current state.
        Remains public due to off-policy learning applications.
        :return:
        """
        action_values = self.value_approximator.get_values(state)

        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(self.num_actions)  # randomly explore
        else:
                greedy_choices = RLHelpers.all_argmax(action_values)  # act greedy
                chosen_action = np.random.choice(greedy_choices)

        return chosen_action # , action_values[chosen_action]

    def _get_policy(self, state):
        # Exploration
        policy = np.ones((self.num_actions))*self.epsilon/self.num_actions

        # Greedy
        action_values = self.value_approximator.get_values(state)
        greedy_choices = RLHelpers.all_argmax(action_values) # technically
        policy[greedy_choices] += (1 - epsilon)/len(greedy_choices)

        assert .999 < policy.sum() < 1.001
        return np.array(policy)

    def start(self, state):
        if self.off_policy_agent is None:
            next_action = self.select_action(state)
        else:
            next_action = self.off_policy_agent.select_action(state) # Yes, I'm using a private method, and may want to expose that again

        self.previous_state = state
        self.previous_action = next_action
        return next_action

    def step(self, reward, state):
        """
        Advance 1 step in the world, and apply self.algorithm update.
        Note: This is built for clarity, rather than efficiency. self.select_action(state) is actually pretty slow
        because it evaluates all possible action_values, which is redundant with some of the operations in this method.
        You could get significant speed-up (1.5-2x) if you optimized this.

        :param reward: Reward value for action taken
        :param state: Current state
        :return: None
        """
        if self.off_policy_agent is None:
            next_action = self.select_action(state)
        else:
            next_action = self.off_policy_agent.select_action(state) # Yes, I'm using a private method, and may want to expose that again
        previous_action_value = self.value_approximator.get_value(self.previous_state, self.previous_action)

        # Update equation
        if self.algorithm == TdControlAlgorithm.QLearner:
            action_values = self.value_approximator.get_values(state)
            optimal_next_value = action_values.max()
            delta = reward + self.gamma * optimal_next_value - previous_action_value
        elif self.algorithm == TdControlAlgorithm.Sarsa:
            next_action_value = self.value_approximator.get_value(state, next_action)
            delta = reward + self.gamma * next_action_value - previous_action_value
        elif self.algorithm == TdControlAlgorithm.ExpectedSarsa:
            # TODO: Did I mess this up...shouldn't policy be elementwise multiplied with all possible action_values?
            action_values = self.value_approximator.get_values(state)
            delta = reward + self.gamma * np.sum(self._get_policy(state) * action_values) - previous_action_value
        else:
            raise Exception("Invalid algorithm selected for TD Control Agent: %s. Select from TdControlAlgorithm enum.")

        if(np.abs(delta*alpha) > 1):
            print(state, " : ", delta*alpha)
        self.value_approximator.update_weights(delta*alpha, self.previous_state, self.previous_action)

        self.previous_state = state
        self.previous_action = next_action
        return next_action

    def end(self, reward):
        """
        Very last step.
        :param reward: Reward gained for action taken
        :return: None
        """
        previous_action_value = self.value_approximator.get_value(self.previous_state, self.previous_action)
        delta =  reward - previous_action_value
        self.value_approximator.update_weights(delta, self.previous_state, self.previous_action)

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
            pickle.dump(self.value_approximator, open(save_path, "wb"))
            print("Agent memory saved to: ", save_path)
        except:
            print("Unable to save agent memory to specified directory: %s " % save_path)

    def load_agent_memory(self, load_path):

        if os.path.isfile(load_path):
            new_value_approximator = pickle.load(open(load_path, "rb"))

            if(self.value_approximator.check_config_match(new_value_approximator)):
                print("Loaded agent memory from: %s" % (load_path))
                self.value_approximator = new_value_approximator
                return True
            else:
                response = input(
                    "Unable to load agent. Previous encoding does not match. " + \
                    "Would you like to DELETE/OVERWRITE previous agent?")
                if response.lower() == "y" or response.lower() == "yes":
                    print("Agent data not loaded. Data will be overwritten at next save.")
                    return False # did not load - going to overwrite
                else:
                    raise Exception("Aborting...agent configuration are not compatible.")
        else:
            print("Warning: Unable to load agent memory. Program will proceed without loading." \
            "\n\tCould not find file: %s" % (load_path))
            return False



if __name__ == "__main__":
    from Agents import HumanAgent
    from FunctionApproximators import TileCodingStateActionApproximator
    import Trainer
    import gym

    ############### Environment Setup (and configuration of agent for env) ###############

    # # Specific to CartPole-v1
    # # ??? pos, vel, ang_pos, ang_vel ???
    # state_limits = np.array([[-2.5, 2.5], [-2.5, 2.5], [-0.3, 0.3], [-1, 1]])
    # feature_resolution = np.array([16,16,16,16])

    # ---Specific to MountainCar-v0---
    # observations = [pos, vel]
    env_name = 'MountainCar-v0'
    state_boundaries = np.array([[-1.2, 0.5], [-0.07, 0.07]])
    tile_resolution = np.array([32, 32])

    env = gym.make(env_name)

    num_tilings = 32
    epsilon = 0.01 # 0.01
    gamma = 1  # discount factor
    alpha = 1/(2**2) # learning rate: .1 to .5 Converges in a few ~1000 episodes down to about -100

    approximator = TileCodingStateActionApproximator.TileCodingStateActionApproximator(
        env_name,
        state_boundaries,
        env.action_space.n,
        tile_resolution = tile_resolution,
        num_tilings = num_tilings)


    ############### Create And Configure Agent ###############
    # Tile Coder Setup
    agent_info = {"num_actions": env.action_space.n,
                  "epsilon": epsilon,
                  "gamma": gamma,
                  "alpha": alpha,
                  "algorithm": TdControlAlgorithm.QLearner,
                  "state_action_value_approximator": approximator,
                  # "off_policy_agent": HumanAgent.HumanAgent(),
                  }

    agent = TDControlAgent(agent_info)
    agent_file_path = os.getcwd() + r"/AgentMemory/Agent_%s_%s.p" % (agent.name, env_name)
    load_status = agent.load_agent_memory(agent_file_path)

    ############### Trainer Setup (load run history) ###############
    trainer_file_path = os.getcwd() + r"/../TrainingHistory/History_%s_%s.p" % (agent.name, env_name)

    trainer = Trainer.Trainer(env, agent)
    if(load_status):
        trainer.load_run_history(trainer_file_path)
    trainer.plot_value_function()

    ############### Define Run inputs and Run ###############
    total_episodes = 50
    max_steps = 1000
    render_interval = 0 # 0 is never

    off_policy_agent = agent_info.get("off_policy_agent", None)
    if  off_policy_agent is not None and off_policy_agent.name == "Human":
        # For human as the off-policy, I'm currently playing 'live', so I have to render, and limit episodes
        total_episodes = 10
        render_interval = 1


    trainer.train_fixed_steps(total_episodes, max_steps, render_interval) # multiple runs for up to total_steps

    ############### Save to file and plot progress ###############
    agent.save_agent_memory(agent_file_path)
    trainer.save_run_history(trainer_file_path)
    Trainer.plot(agent, np.array(trainer.rewards) )
