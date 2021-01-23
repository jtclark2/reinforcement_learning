"""
Still getting my head around all the different agents. I fully expect to refactor as I build different agents, but
I'm not sure what methods/hierarchy I'll be using yet.

TODO:
 - Dyna Q and Dyna Q +
 - Improve initialization
"""

import numpy as np
from enum import Enum

# Relative imports
from ToolKit import RLHelpers
import pickle
import os

class TdControlAlgorithm(Enum):
    QLearner = 1
    Sarsa = 2
    ExpectedSarsa = 3

class SemiGradientTdAgent:
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
        # None is perfect for most simulations - replace with model for slower environments, or real-world (robotics)
        self.model = agent_info.get("model", None)

        # Hyperparameters
        self.epsilon = agent_info.get("epsilon", 0.01) # In case of epsilon greedy exploration
        self.gamma = agent_info.get("gamma", 1) # discount factor
        self.alpha = agent_info["alpha"] # step size of learning rate


        self.algorithm = agent_info.get("algorithm", TdControlAlgorithm.QLearner)
        assert isinstance(self.algorithm, TdControlAlgorithm)
        self.name = self.algorithm.name
        self.off_policy_agent = agent_info.get("off_policy_agent", None)
        # This can work with SARSA and Expected SARSA, but then you really should use importance sampling, which
        # is not a priority for this project, because:
        #   1) The agent interface I use for off-policy action selection does not expose the policy directly.
        #   We could build up an model for the policy, but that is far more effort than this application merits.
        #   2) This project is intended to practice practical skills, and that's neither a concept that i need to
        #   solidify, nor a frequently used/practical approach to gain experience implementing.
        # assert self.algorithm == TdControlAlgorithm.QLearner

    def reset(self, agent_info={}):
        """
        Setup for the agent at beginning of episode.
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

        # Select next action
        if self.off_policy_agent is None:
            next_action = self.select_action(state)
        else:
            next_action = self.off_policy_agent.select_action(state) # Yes, I'm using a private method, and may want to expose that again

        self._update_q(self.previous_state, self.previous_action, reward, state, next_action)

        if self.model is not None:
            self.model.record_transition(self.previous_state, self.previous_action, reward, state)
            model.simulate(self._update_q) # learn by simulating previous experience

        self.previous_state = state
        self.previous_action = next_action
        return next_action

    def _update_q(self, previous_state, previous_action, reward, state, next_action):
        """
        Update Q function approximation based on (SARSA, Expected SARSA, Q Learning), using TD update.
        Based on a previous_state S, and previous_action A, the environment updates, creating reward R. The environment
        then transitions to state S'. I think of this as the current state, because as far as the environment is
        concerned, that's what it is. Sometimes the literature (confusingly) calls this next_state, to differentiate
        it from the previous one.
        This is where the algorithms diverge.
            - Sarsa selects the next_action, A', which is used to estimate the new state-action-value, Q(S',A')
            - Q Learning considers all possible next actions, and using the optimal one (without needing to decide on
                the next action yet.
            -  Expected Sarsa averages over all possible actions in the ratio the policy dictates.
        :param next_action: The next action selected (but not yet taken). Only needed for SARSA.
        :param reward: The reward gained based on the previous environment step.
        :param state: The state the environment has just transitioned into. This is technically next_state, since Q updates technically happen after action selection, but
        before arrival to the next state. this is technically the next state that we we will transition to.

        :return:
        """
        previous_action_value = self.value_approximator.get_value(previous_state, previous_action)
        # Update equation
        if self.algorithm == TdControlAlgorithm.QLearner:
            action_values = self.value_approximator.get_values(state)
            optimal_next_value = action_values.max()
            delta = reward + self.gamma * optimal_next_value - previous_action_value
        elif self.algorithm == TdControlAlgorithm.Sarsa:
            next_action_value = self.value_approximator.get_value(state, next_action)
            delta = reward + self.gamma * next_action_value - previous_action_value
        elif self.algorithm == TdControlAlgorithm.ExpectedSarsa:
            action_values = self.value_approximator.get_values(state)
            delta = reward + self.gamma * np.sum(self._get_policy(state) * action_values) - previous_action_value
        else:
            raise Exception("Invalid algorithm selected for TD Control Agent: %s. Select from TdControlAlgorithm enum.")
        self.value_approximator.update_weights(delta * self.alpha, previous_state, previous_action)

        # EXPERIMENTAL correction to semi-gradient descent, making it true gradient descent
        # It seems to work - still converges, and matches Monte Carlo better this way. That said, neither this form
        # or the MC form match my expectations. Book has a linear plot, which this is not.
        # More investigation would be needed.
        # See further discussion here:
        # https://stats.stackexchange.com/questions/347295/why-semi-gradient-is-used-instead-of-the-true-gradient-in-q-learning
        FULL_GRADIENT_EXPERIMENT = False
        if FULL_GRADIENT_EXPERIMENT and next_action is None:
            next_action = self.select_action(state)
        self.value_approximator.update_weights(-delta * self.alpha, state, next_action)
        # End of Experimental

    def end(self, reward):
        """
        Very last step.
        Note that this is very similar to _update_q, but a little simpler, since state and action are not needed.
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
    from FunctionApproximators import TileCodingStateActionApproximator
    from Models import DeterministicModel
    import Trainer
    import gym

    ############### Environment Setup (and configuration of agent for env) ###############


    # Specific to CartPole-v1
    # ??? pos, vel, ang_pos, ang_vel ???
    # really favors more exploration epsilon >= 0.1...lower epsilon takes forever to learn, and higher never stabilizes
    # alpha can be as high as 1 and stay fairly stable, though there is no need to push it
    # env_name = 'CartPole-v1'
    # state_boundaries = np.array([[-2.5, 2.5], [-2.5, 2.5], [-0.3, 0.3], [-1, 1]])
    # tile_resolution = np.array([16,16,16,16])
    # num_tilings = 32
    # env = gym.make(env_name)
    # epsilon = 0.1 # 0.01
    # gamma = 1  # discount factor
    # alpha = 1/(2**1) # learning rate: .1 to .5 Converges in a few ~1000 episodes down to about -100

    # # ---Specific to MountainCar-v0---
    # # observations = [pos, vel]
    # env_name = 'MountainCar-v0'
    # state_boundaries = np.array([[-1.2, 0.5], [-0.07, 0.07]])
    # tile_resolution = np.array([32, 32])
    # num_tilings = 32
    # env = gym.make(env_name)
    # epsilon = 0.1 # 0.01
    # gamma = 1  # discount factor
    # alpha = 1/(2**1) # learning rate: .1 to .5 Converges in a few ~1000 episodes down to about -100
    # model = DeterministicModel.DeterministicModel(5)


    # Specific to RandomWalk (a custom pseudo-env I created)
    from Environments import RandomWalkEnv
    env_name = 'RandomWalk_v0'
    state_boundaries = np.array([[0,1000]])
    tile_resolution = np.array([10])
    env = RandomWalkEnv.RandomWalkEnv()
    num_tilings = 1
    epsilon = 1 # 1 # 0.01
    gamma = 1  # discount factor
    alpha = 1/(2**5) # learning rate: .1 to .5 Converges in a few ~1000 episodes down to about -100
    model =  None

    approximator = TileCodingStateActionApproximator.TileCodingStateActionApproximator(
        env_name,
        state_boundaries,
        env.action_space.n,
        tile_resolution = tile_resolution,
        num_tilings = num_tilings)

    model = None # DeterministicModel.DeterministicModel(5)


    ############### Create And Configure Agent ###############
    # Tile Coder Setup
    agent_info = {"num_actions": env.action_space.n,
                  "epsilon": epsilon,
                  "gamma": gamma,
                  "alpha": alpha,
                  "algorithm": TdControlAlgorithm.Sarsa,
                  "state_action_value_approximator": approximator,
                  # "off_policy_agent": HumanAgent.HumanAgent(),
                  "model": model
                  }

    agent = SemiGradientTdAgent(agent_info)
    agent_file_path = os.getcwd() + r"/AgentMemory/Agent_%s_%s.p" % (agent.name, env_name)
    load_status = agent.load_agent_memory(agent_file_path)

    ############### Trainer Setup (load run history) ###############
    trainer_file_path = os.getcwd() + r"/../TrainingHistory/History_%s_%s.p" % (agent.name, env_name)

    trainer = Trainer.Trainer(env, agent)
    if(load_status):
        trainer.load_run_history(trainer_file_path)

    ############### Define Run inputs and Run ###############
    total_episodes = 1000
    max_steps = 1000
    render_interval = 0 # 0 is never

    off_policy_agent = agent_info.get("off_policy_agent", None)
    if  off_policy_agent is not None and off_policy_agent.name == "Human":
        # For human as the off-policy, I'm currently playing 'live', so I have to render, and limit episodes
        total_episodes = 10
        render_interval = 1

    trainer.run_multiple_episodes(total_episodes, max_steps, render_interval) # multiple runs for up to total_steps

    ############### Save to file and plot progress ###############
    agent.save_agent_memory(agent_file_path)
    trainer.save_run_history(trainer_file_path)
    Trainer.plot(agent, np.array(trainer.rewards) )
    # TODO: extend plot_value_function to allow direct input, currently assumes observation space is 2D (eg: pos, vel)
    # trainer.plot_value_function()

    import Trainer
    if env_name == 'RandomWalk_v0':
        x = [x for x in range(1000)]
        y_estimate = [np.average(agent.value_approximator.get_values(np.array([x]))) for x in range(1000)]
        y_actual = [ (x-500)/500 for x in range(1000)]
        Trainer.multiline_plot(x, y_estimate, y_actual)