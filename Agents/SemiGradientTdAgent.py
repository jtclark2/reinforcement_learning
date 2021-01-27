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
        # Off-policy and simulation can work with SARSA and Expected SARSA, but then you really should use importance
        # # sampling, which is not a priority for this project, because:
        #   1) The agent interface I use for off-policy action selection does not expose the policy directly.
        #   We could build up an model for the policy, but that is far more effort than this application merits.
        #   2) This project is intended to practice practical skills, and that's neither a concept that i need to
        #   solidify, nor a frequently used/practical approach to gain experience implementing.
        if self.algorithm != TdControlAlgorithm.QLearner and \
                (self.off_policy_agent is not None or self.model is not None):
            raise Exception("'Off_policy_agent' and 'model' should only be used ith 'algorithm' = QLearner.")

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
        policy[greedy_choices] += (1 - self.epsilon)/len(greedy_choices)

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
            self.model.simulate(self._update_q) # learn by simulating previous experience

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
        # TODO: implement average reward
        # TODO: Implement that other normalization of deviation thing, is that what REINFORCE is?
        # delta -= self.average_reward
        # self.average_reward = self.average_reward + self.beta*delta
        self.value_approximator.update_weights(delta * self.alpha, previous_state, previous_action)

        # EXPERIMENTAL correction to semi-gradient descent, making it true gradient descent
        # Basically, semigrad approximates deltaError = (Ut-v(St,w))(d(Ut)/dw - dv(St,w)/dw) by dropping the d(Ut) term.
        # This experimental correction adds it back in. ExperimentalCorrection = (Ut-v(St,w))(d(Ut)/dw)
        # where U = R' + gamma*v_hat(S', w) = R' + gamma*d(v_hat)/dw -->  -d(U)/dw = gamma*x,
        # and we already calculated delta = (Ut-v(St,w))
        # Correction = (Ut-v(St,w))(gamma * d(v_hat)/dw) = -delta * gamma  ... the derivative is applied in the
        # value approximator's update, and the alpha is the same step size used on the original update
        # It seems to work - still converges, and matches Monte Carlo better this way. That said, neither this form
        # or the MC form match my expectations. Book has a linear plot, which this is not.
        # More investigation would be needed.
        # See further discussion here:
        # https://stats.stackexchange.com/questions/347295/why-semi-gradient-is-used-instead-of-the-true-gradient-in-q-learning
        FULL_GRADIENT_EXPERIMENT = False
        if FULL_GRADIENT_EXPERIMENT and next_action is None:
            next_action = self.select_action(state)
            self.value_approximator.update_weights(-delta * self.alpha * self.gamma, state, next_action)
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
    from Trainers import GymTrainer
    from ToolKit.PlottingTools import PlottingTools
    from Agents.EnergyPumpingMountainCarAgent import EnergyPumpingMountainCarAgent
    import gym

    def setup_and_train(config, total_episodes = 100, render_interval = 0, load = False, save = False):
        # Unpack config
        state_boundaries = config['state_boundaries']
        env = config['env']
        env_name = config['env_name']
        ### Hyperparameters ###
        tile_resolution = config['tile_resolution']
        num_tilings = config['num_tilings']
        epsilon = config['epsilon']
        gamma = config['gamma']
        alpha = config['alpha']
        model = config['model']
        off_policy_agent = config['off_policy_agent']
        algorithm = config['algorithm']
        initial_value = config['initial_value']

        ############### Create And Configure Agent ###############
        approximator = TileCodingStateActionApproximator.TileCodingStateActionApproximator(
            env_name=env_name,
            state_boundaries=state_boundaries,
            num_actions=env.action_space.n,
            tile_resolution=tile_resolution,
            num_tilings=num_tilings,
            initial_value=initial_value)

        agent_info = {"num_actions": env.action_space.n,
                      "epsilon": epsilon,
                      "gamma": gamma,
                      "alpha": alpha,
                      "algorithm": algorithm,
                      "state_action_value_approximator": approximator,
                      "off_policy_agent": off_policy_agent,
                      "model": model
                      }
        agent = SemiGradientTdAgent(agent_info)

        ############### Create Trainer ###############
        trainer = GymTrainer.GymTrainer(env, agent)

        ############### Load prior experience/history ###############
        agent_file_path = os.getcwd() + r"/AgentMemory/Agent_%s_%s.p" % (agent.name, env_name)
        trainer_file_path = os.getcwd() + r"/../TrainingHistory/History_%s_%s.p" % (agent.name, env_name)
        if load:

            load_status = agent.load_agent_memory(agent_file_path)
            if (load_status):
                trainer.load_run_history(trainer_file_path)

        ############### Train ###############
        convergent_reward = trainer.run_multiple_episodes(total_episodes, render_interval)  # multiple runs for up to total_steps

        ############### Save to file and plot progress ###############
        if save:
            agent.save_agent_memory(agent_file_path)
            trainer.save_run_history(trainer_file_path)

        return agent, trainer

    def get_mountain_car_configuration():
        config = {}

        ### Intrinsic to MountainCar-v0 ###
        # observations = [pos, vel]
        env_name = 'MountainCar-v0'
        # In theory, env.observation_space.high and env.observation_space.low will return these; however, they allow
        # inf for values they don't really care about, and that wreaks havoc on the tile mapper, which needs bounds
        # fully defined
        config['state_boundaries'] = np.array([[-1.2, 0.5], [-0.07, 0.07]])
        config['env'] = gym.make(env_name)
        config['env_name'] = env_name

        ### Hyperparameters ###
        # 8 to 32 is reasonable. > 32 adds very little, and gets expensive
        config['tile_resolution'] = np.array([32, 32])
        config['num_tilings'] = 32
        config['epsilon'] = 0.1  # between 0 - 0.1 is pretty reasonable for this problem
        config['gamma'] = 1  # discount factor - pretty much always 1 for this problem, since it's episodic
        config['alpha'] = 1 / (2 ** 1)  # learning rate: anywhere from 1 (high, but it works) to 1/16 work well
        config['model'] = DeterministicModel.DeterministicModel(5)  # 5 is reasonably fast. None is also reasonable
        config['off_policy_agent'] = None
        config['algorithm'] = TdControlAlgorithm.QLearner
        config['initial_value'] = 0.0

        return config

    def get_cart_pole_config():
        config = {}

        ### Intrinsic to CartPole-v1 ###
        # observations = [pos, vel, ang_pos, ang_vel]
        env_name = 'CartPole-v1'
        # In theory, env.observation_space.high and env.observation_space.low will return these; however, they allow
        # inf for values they don't really care about, and that wreaks havoc on the tile mapper, which needs bounds
        # fully defined
        config['state_boundaries'] = np.array([[-2.5, 2.5], [-2.5, 2.5], [-0.3, 0.3], [-1, 1]])
        config['env'] = gym.make(env_name)
        config['env_name'] = env_name

        ### Hyperparameters ###
        config['tile_resolution'] = np.array([16,16,16,16])
        config['num_tilings'] = 32
        config['epsilon'] = 0.1  # between 0 - 0.1 is pretty reasonable for this problem
        config['gamma'] = 1  # discount factor - pretty much always 1 for this problem, since it's episodic
        config['alpha'] = 1 / (2 ** 1)  # learning rate: anywhere from 1/2 (high, but it works) to 1/32 work well
        config['model'] = DeterministicModel.DeterministicModel(5)  # 5 is reasonably fast. None is also reasonable
        config['off_policy_agent'] = None
        config['algorithm'] = TdControlAlgorithm.QLearner
        config['initial_value'] = 0.0

        return config

    def get_random_walk_config():
        from Environments import RandomWalkEnv
        config = {}

        ### Intrinsic to RandomWalk_v0 ###
        # observations = [pos]
        env_name = 'RandomWalk_v0'
        # In theory, env.observation_space.high and env.observation_space.low will return these; however, they allow
        # inf for values they don't really care about, and that wreaks havoc on the tile mapper, which needs bounds
        # fully defined
        config['state_boundaries'] = np.array([[0,1000]])
        config['env'] = RandomWalkEnv.RandomWalkEnv()
        config['env_name'] = env_name

        ### Hyperparameters ###
        config['tile_resolution'] = np.array([10])
        config['num_tilings'] = 1
        config['epsilon'] = 1  # For the true random walk, 1. Obviously, it's a bit silly. It's more of a diagnostic.
        config['gamma'] = 1  # discount factor - pretty much always 1 for this problem, since it's episodic
        config['alpha'] = 1 / (2 ** 5)  # learning rate
        config['model'] = None
        config['off_policy_agent'] = None
        config['algorithm'] = TdControlAlgorithm.QLearner
        config['initial_value'] = 0.0

        return config


    """
    A bit of informal testing, though I'm not writing true unit testing for this - just a very basic sanity check.
    """
    def test__mountain_car_q_learner():
        """
        Purpose:
            1) Test running with mountain car env. Test dimension matching in 2D observation space
            2) Test convergence in sparsely reward env (initialization is 0, optimistic for this problem).
            3) Test Q-learner algorithm, with model simulation.
        :return: None
        """
        total_episodes = 60
        render_interval = 0 # 0 is never
        config = get_mountain_car_configuration()
        config['tile_resolution'] = np.array([8, 8])   # Just speeding things up a bit
        config['num_tilings'] = 8
        config['alpha'] = 1/4
        agent, trainer = setup_and_train(config, total_episodes, render_interval)
        convergent_reward = np.average(trainer.rewards[-20:])
        assert convergent_reward > -180 # I've seen about -160 on ave
        print(f"test__mountain_car_q_learner converged to {convergent_reward} in {len(trainer.rewards)} episodes!")

    def test__expected_sarsa_cart_pole():
        """
        Purpose:
            1) Test Cart Pole Environment, which has 4D observation space.
            2) Test Expected Sarsa algorithm
            3) Test Agent Save and Load

        Future improvement (to Trainer, not this module): Measure progress in steps, rather than episodes, for more
        consistent pace of learning.
        :return: None
        """
        total_episodes = 100
        render_interval = 0  # 0 is never
        thresh = 140
        config = get_cart_pole_config()
        config['algorithm'] = TdControlAlgorithm.ExpectedSarsa
        config['model'] = None

        _, trainer = setup_and_train(config, total_episodes, render_interval, load=False, save=True)
        convergent_reward = np.average(trainer.rewards[-20:])
        for i in range(1,10): # Happy to give it up to 1000 steps to converge, but it takes forever, so we can check early
            _, trainer = setup_and_train(config, total_episodes, render_interval, load=True, save=True)
            convergent_reward = np.average(trainer.rewards[-20:])
            if convergent_reward > thresh:
                break

        assert convergent_reward > thresh
        print(f"test__expected_sarsa_cart_pole converged to {convergent_reward} in {len(trainer.rewards)} episodes!")

    def test__sarsa_walk():
        """
        Purpose:
            1) Test in 1D RandomWalk Env.
            2) Test Sarsa algorithm
        :return:
        """
        total_episodes = 200
        render_interval = 0  # 0 is never
        config = get_random_walk_config()
        config['algorithm'] = TdControlAlgorithm.Sarsa
        config['model'] = None
        config['epsilon'] = 0.5
        agent, trainer = setup_and_train(config, total_episodes, render_interval)
        convergent_reward = np.average(trainer.rewards[-20:])
        assert convergent_reward > .8
        print(f"test__sarsa_walk converged to {convergent_reward} in {len(trainer.rewards)} episodes!")

    def test__off_policy_q_learner():
        """
        Purpose:
            1) Test off-policy learning.
        Note (kind of cool):
            I've seen the best performance with policies that finish, but aren't that great
            (eg: EnergyPumpingMountainCarAgent with epsilon = 0.5).
            Great policies start to form a great value-function, but it's so noisy that the agent sort of falls off the
            good path. It still speeds up learning, just not as well, due to all the mis-steps.
            Really bad examples, are speed up the process by showing what not to do. However, the search space is small,
            so the value function is relatively unexplored near the goal, and lots of exploration is still needed.
            I can draw lots of analogies to trying to learn from an expert and missing things along the way. If you
            go over it enough, you'll learn it, but it can be useful to experience it yourself to explore questions, and
            to solidify those gaps in knowledge. Similarly, learning alongside someone who doesn't
            know anymore than you do speeds things up a bit, but your both making similar mistakes.
        :return: None
        """
        train_episodes = 100
        test_episodes = 20
        render_interval = 0 # 0 is never
        config = get_mountain_car_configuration()
        config['tile_resolution'] = np.array([8, 8])   # Just speeding things up a bit
        config['num_tilings'] = 8
        config['alpha'] = 1/4
        config['off_policy_agent'] = EnergyPumpingMountainCarAgent()
        config['epsilon'] = 0.0 # do your best!
        # initialize pessimistically...discourages exploration, but encourages mimicking off-policy approach
        # config['initial_value'] = 0
        agent, trainer = setup_and_train(config, train_episodes, render_interval, save=True) # Train off-policy
        PlottingTools.plot_action_value_2d(agent.value_approximator)

        config['off_policy_agent'] = None
        render_interval = 20 # 0 is never
        agent, trainer = setup_and_train(config, test_episodes, render_interval, load=True) # Run for test
        convergent_reward = np.average(trainer.rewards[-10:])
        assert convergent_reward > -170 # I've seen about -150 on ave
        print(f"test__off_policy_q_learner converged to {convergent_reward} in {len(trainer.rewards)} episodes!")
        PlottingTools.plot_action_value_2d(agent.value_approximator)


    def tests__all_semi_gradient():
        test__off_policy_q_learner
        test__mountain_car_q_learner()
        test__expected_sarsa_cart_pole()
        test__sarsa_walk()
        print("All SemiGradientTdAgent tests passed!")

    ############### Environment Setup (and configuration of agent for env) ###############
    total_episodes = 300
    render_interval = 0 # 0 is never

    env_selection = "Test"
    mode = "Test" # "Test", "Manual", "Normal"
    plot = True
    configs = {"mountain_car": get_mountain_car_configuration,
               "cart_pole": get_cart_pole_config,
               "random_walk": get_random_walk_config}

    if mode == "Test":
        tests__all_semi_gradient()
    else:
        config = configs[env_selection]()
        agent, trainer = setup_and_train(config, total_episodes, render_interval)

        if plot:
            PlottingTools.plot_smooth(trainer.rewards)

            if env_selection == "mountain_car":
                PlottingTools.plot_action_value_2d(agent.value_approximator)

            if env_selection == "random_walk":
                x = [x for x in range(1000)]
                y_estimate = [np.average(agent.value_approximator.get_values(np.array([x]))) for x in range(1000)]
                y_actual = [ (x-500)/500 for x in range(1000)]
                PlottingTools.multiline_plot(x, y_estimate, y_actual)