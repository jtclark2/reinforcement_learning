from Agents import KeyBoardInput
import numpy as np

class HumanAgent:
    """
    Just an interface to the keyboard, so the human can be an agent too.
    """
    def __init__(self, agent_info={}):
        pass

    def reset(self, agent_info={}):
        pass

    def start(self, state):
        """
        Kick off a listening thread for keyboard input. Different from previous tries, because I want to capture
        input between frames.
        :param state:
        :return:
        """
        self.keyboard = KeyBoardInput.KeyBoardInput()
        action = self.keyboard.get_action()
        return action

    def step(self, reward, state):
        """
        Advance 1 step in the world, using SARSA update.

        :param reward: Reward value for action taken
        :param state: Current state
        :return: None
        """
        #keyboard input...
        action = self.keyboard.get_action()
        return action

    def end(self, reward):
        """
        Very last step.
        :param reward: Reward gained for action taken
        :return: None
        """
        action = self.keyboard.get_action()
        return action

    def message(self):
        pass

    def save_agent_memory(self, save_path):
        pass

    def load_agent_memory(self, load_path):
        pass

if __name__ == "__main__":
    from Agents import Optimizers
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
    alpha = 1/(2**0) # learning rate: .1 to .5 Converges in a few ~1000 episodes down to about -100

    approximator = TileCodingStateActionApproximator.TileCodingStateActionApproximator(
        state_boundaries,
        env.action_space.n,
        tile_resolution = tile_resolution,
        num_tilings = num_tilings)


    ############### Create And Configure Agent ###############
    # Tile Coder Setup

    env = gym.make(env_name)
    agent_info = None
    agent = HumanAgent(agent_info)
    agent_file_path = None
    load_status = agent.load_agent_memory(agent_file_path)

    ############### Trainer Setup (load run history) ###############
    trainer_file_path = None

    trainer = Trainer.Trainer(env, agent)
    if(load_status):
        trainer.load_run_history(trainer_file_path)

    ############### Define Run inputs and Run ###############
    total_episodes = 1
    max_steps = 1000
    render_interval = 1 # # I better be able to see if I'm playing
    trainer.train_fixed_steps(total_episodes, max_steps, render_interval) # multiple runs for up to total_steps

    ############### Save to file and plot progress ###############
    agent.save_agent_memory(agent_file_path)
    trainer.save_run_history(trainer_file_path)
    Trainer.plot(agent, np.array(trainer.rewards) )
