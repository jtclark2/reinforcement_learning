from ToolKit import RLHelpers
import pickle
import numpy as np
import os

class MonteCarloQAgent:
    def __init__(self, agent_info={}):
        """
        Monty Carlo methods DO NOT LEARN if they do not terminate. Therefore, this algorithm can get stuck...
        and it does ALL the time. This video doesn't focus on this aspect, but it does note it:
        https://www.coursera.org/learn/sample-based-learning-methods/lecture/RZeRQ/sarsa-in-the-windy-grid-world

        How to overcome not finishing (ie: not knowing if you did better or worse than other times):
        1) Exploring Starts (not used): Only works if you have enough control over the environment to start in arbitrary/random
        states.
        2) Off-Policy learning (not used): Follow a known policy (even a suboptimal one), as long as it finishes a descent chunk of the
        time. However, needing another policy is a big drawback to begin with. Also, you should then use importance
        sampling, which requires knowledge of the policy you are sampling from. That means either having the policy,
        or building a sampled model (eg: if you're using playback). I played with it a bit (ignoring importance
        sampling) just to convince myself it works. If you're policy is close-ish to optimal, then it works well enough.
        3) Intermediate Rewards : Give it intermediate rewards, such as how far towards the goal it has traveled. This
        works reasonably well, but introduces risk of perverse incentives emerging. In other words, it may find a way to
        optimize the reward which does not line up with the ultimate objective.
            In order for this to improve in a stable way, gamma < 1. I think this is because the system can wander
            significantly before reaching the end of it's run, and it doesn't always know which of that wandering was
            helpful. By utilizing the discount factor, you can decrease the impact of some of that wandering. In the
            tabular form of this problem (I've written this more generally) you would ignore any loops in the path that
            cycle back to a previously encountered state. The discount factor is less precise, but has a similar effect.

        On top of all that, it learns much slower than Q-learning, both because of the information that is being ignored
        with respect to adjacent state-values, and because it's a higher variance algorithm, so I had to low alpha
        significantly (by 8x).

        In conclusion, great tool as a thought experiment, but I can't think of any real-world application where
        it's performance comes close to TD learners, such as Q-Learning and SARSA.

        :param agent_info:
        """
        # Defined by the environment
        self.num_actions = agent_info["num_actions"] # number of available actions; must be defined

        # State Dependant Variables
        self.previous_action = None
        self.previous_state = None

        # Encoding (this has been encapsulated in it's own class, for dependency inversion)
        self.value_approximator = agent_info["state_action_value_approximator"] # no default value ( we want to fail here if encoder is not provided )

        # Hyperparameters
        self.epsilon = agent_info.get("epsilon", 0.01) # In case of epsilon greedy exploration
        self.gamma = agent_info.get("gamma", 1) # discount factor
        self.alpha = agent_info["alpha"] # step size of learning rate

        self.name = "MonteCarloQ"
        self.run_history = []


    def reset(self, agent_info={}):
        """
        Setup for the agent called when the experiment first starts.
        :return: None
        """
        self.run_history = []
        # TODO: Bits of reward shaping are in here. Ultimately, I want to allow this, but I definitely don't want it
        # hardcoded in like this. This makes the algorithm very specific to certain problems
        # self.best_yet = -10000000 # DELETE - Reward Shaping

    def select_action(self, state):
        """
        Selects the next action, based on the current state.
        Remains public due to off-policy learning applications.
        :return:
        """
        action_values = self.value_approximator.get_values(state)

        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(self.num_actions)  # randomly explore

            # The next 2 lines are cheating - they are specific to mountain car, as pseudo off-policy learning
            # if self.previous_action is None: self.previous_action = 0
            # chosen_action = self.previous_action
        else:
                greedy_choices = RLHelpers.all_argmax(action_values)  # act greedy
                chosen_action = np.random.choice(greedy_choices)

        return chosen_action

    def start(self, state):
        next_action = self.select_action(state)

        self.previous_state = state
        self.previous_action = next_action
        return next_action

    def step(self, reward, state):
        next_action = self.select_action(state)
        self.run_history.append((reward, self.previous_state, self.previous_action))

        # self.best_yet = np.max([self.best_yet, state[0]]) # DELETE - Reward Shaping

        self.previous_state = state
        self.previous_action = next_action
        return next_action

    def end(self, reward):
        # DELETE - Reward shaping - this is an example of reward shaping. It's not great, because it's specific to a single
        # problem (mountain car in this case).
        # if len(self.run_history) == 200:
        #     reward += (self.best_yet-0.5)*100
        #     # reward += self.previous_state[0]*100 - 200
        self.run_history.append((reward, self.previous_state, self.previous_action))

        g = 0
        self.run_history.reverse()
        i = 0
        for r, state, action in self.run_history:
            i += 1
            # r, state, action = step
            g = r + self.gamma * g
            delta = self.alpha*(g - self.value_approximator.get_value(state, action))
            # Error function and gradient and bundled into the approximator
            self.value_approximator.update_weights(delta, state, action)

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
    from Agents import MonteCarloQAgent
    from FunctionApproximators import TileCodingStateActionApproximator
    import Trainer

    ############### Environment Setup (and configuration of agent for env) ###############

    # # Specific to CartPole-v1
    # # ??? pos, vel, ang_pos, ang_vel ???
    # state_limits = np.array([[-2.5, 2.5], [-2.5, 2.5], [-0.3, 0.3], [-1, 1]])
    # feature_resolution = np.array([16,16,16,16])

    # # ---Specific to MountainCar-v0---
    # # observations = [pos, vel]
    # env_name = 'MountainCar-v0'
    # state_boundaries = np.array([[-1.2, 0.5], [-0.07, 0.07]])
    # tile_resolution = np.array([16, 16])
    # env = gym.make(env_name)
    # num_tilings = 16
    # epsilon = 0.01 # 0.01
    # gamma = .98  # discount factor
    # alpha = 1/(2**4) # learning rate

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


    ############### Create And Configure Agent ###############
    # Tile Coder Setup
    agent_info = {"num_actions": env.action_space.n,
                  "epsilon": epsilon,
                  "gamma": gamma,
                  "alpha": alpha,
                  "state_action_value_approximator": approximator,
                  }

    agent = MonteCarloQAgent.MonteCarloQAgent(agent_info)
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
    frame_delay = 0.01

    trainer.run_multiple_episodes(total_episodes, max_steps, render_interval, frame_delay) # multiple runs for up to total_steps

    ############### Save to file and plot progress ###############
    agent.save_agent_memory(agent_file_path)
    trainer.save_run_history(trainer_file_path)
    Trainer.plot(agent, np.array(trainer.rewards))
    # trainer.plot_value_function()

    import Trainer
    if env_name == 'RandomWalk_v0':
        x = [x for x in range(1000)]
        y_estimate = [np.average(agent.value_approximator.get_values(np.array([x]))) for x in range(1000)]
        y_actual = [ (x-500)/500 for x in range(1000)]
        Trainer.multiline_plot(x, y_estimate, y_actual)
