import time
from datetime import datetime
import pickle
import os
import numpy as np
from ToolKit.PlottingTools import PlottingTools

# TODO: Pass in a reward shaping function...I know it's frowned upon, beause we all want fully generalized AI,
# but it is a practical tool that should not be overlooked
# TODO: Formalize interfaces / abstract classes
class GymTrainer():
    """
    This class works with Gym environment to train agents.
    """

    def __init__(self, env, agent):
        self.rewards = []
        self.env = env
        self.observation = env.reset()
        self.agent = agent
        self.episode_count = 0
        self.plotter = PlottingTools()

    def run_multiple_episodes(self, target_steps, render_interval, frame_delay=0.01, history_file_path=None, save=False ):
        """
        This may need to rethinking with regard to termination condition...we may want more flexibility than a fixed
        number of steps in more advanced situations.

        :param target_steps: Total planned steps to be taken. Run will be allowed to complete if it starts before
            reaching this value.
        :param render_interval: Render will occur once every render_interval episodes

        :return: None
        """
        step_count = 0
        greatest_reward = float("-inf")
        start_time = datetime.now()
        self.smoothing = 10
        while(step_count < target_steps):
            if render_interval != 0 and self.episode_count % render_interval == 0: # Render every n episodes
                print(f"Render ON. Attempt: {self.episode_count} Greatest Reward so far: {greatest_reward}")
                render = True
            else:
                render = False

            self.episode_count += 1
            step_increment, reward = self.run_episode(render, frame_delay)
            self.rewards.append(reward)
            greatest_reward = max(greatest_reward, reward)
            step_count += step_increment
            if self.episode_count % 1 == 0: # TODO: this won't work anymore
                average_reward = np.average(self.rewards[len(self.rewards)-self.smoothing:])
                print("Episode: %d, Steps: %d/%d, Reward: %f, Best: %f, Average: %f" % (self.episode_count, step_count, target_steps, reward, greatest_reward, average_reward))
                # end_time = time.time()
                PlottingTools.plot_smooth(self.rewards, silent=True)
                # start_time, end_time = end_time, time.time()
                # print(f"Plotting took: {end_time-start_time} seconds")
                if save:
                    self.save_run_history(history_file_path) # TODO: pas in save arg, this should be conditional
                    # start_time, end_time = end_time, time.time()
                    # print(f"Saving took: {end_time-start_time} seconds")
                current_time = datetime.now()
                time_per_step = (current_time-start_time)/step_count
                steps_remaining = target_steps-step_count
                time_remaining = steps_remaining*time_per_step

                print(f"Current Time: {current_time}...Projected Completion Time: {current_time + time_remaining}... Time remaining: {time_remaining} seconds")

        print("###################### TRAINING COMPLETE ######################")

    def run_episode(self, render, frame_delay):
        """
        Initializes env, and runs until done.

        :param env: Gym environment.
        :param agent: AgentMemory/acting agent.
        :param max_steps: Max number of steps to run before returning (even if agent is not done)
        :param render: Boolean for whether visuals will render. Env will run MUCH faster if False.
        :param sleep_time: Use to slow playback. Useful for viewing/analyzing performance.

        :return: Tuple: (Time taken, total reward achieved)
        """

        self.observation = self.env.reset()
        self.agent.reset()
        total_reward = 0
        action = self.agent.start(self.observation)
        step_count = 0
        while True:
            step_count += 1
            if render:
                time.sleep(frame_delay)
                self.env.render()
            self.observation, reward, done, info = self.env.step(action)  # take a random action
            total_reward += reward

            if (done is True):
                self.agent.end(reward)
                break
            else:
                action = self.agent.step(reward, self.observation)

        self.env.close()
        return step_count, total_reward

    def save_run_history(self, save_path=""):
        try:
            pickle.dump(self.rewards, open(save_path, "wb"))
            print("Saved training history to: ", save_path)
        except:
            print("Unable to save run history to: ", save_path)

    def load_run_history(self, load_path=""):
        if os.path.isfile(load_path):
            self.rewards = pickle.load(open(load_path, "rb"))
            self.episode_count = len(self.rewards)
            print("Loaded run history. Current episode count: ", len(self.rewards) )
        else:
            print("Warning: Unable to load training_history. Program will proceed without loading.")
            time.sleep(2)

if __name__ == "__main__":
    import gym
    from Agents import HumanAgent, InertAgent, NNAgent
    from ToolKit.PlottingTools import PlottingTools

    ############### Environment Setup (and configuration of agent for env) ###############
    # env_name = 'MountainCar-v0' # 'MountainCar-v0', 'Breakout-v0', 'Breakout-ram-v0', etc.
    env_name = 'LunarLander-v2'

    history_file_path = os.getcwd() + "/TrainingHistory/" + env_name
    env = gym.make(env_name)

    ############### Instantiate and Configure Agent ###############
    # agent = HumanAgent.HumanAgent({"env":env})
    # agent = InertAgent.InertAgent()
    agent_parameters = {
        'network_config': {
            'state_dim': 8,
            'num_hidden_units': 256,
            'num_actions': 4
        },
        'optimizer_config': {
            'step_size': 1e-3,
            'beta_m': 0.9,
            'beta_v': 0.999,
            'epsilon': 1e-8
        },
        'replay_buffer_size': 50000,
        'minibatch_sz': 8,
        'num_replay_updates_per_step': 4,
        'gamma': 0.99,
        'tau': 0.001
    }
    agent = NNAgent.Agent(agent_parameters)

    ############### Trainer Setup (load run history) ###############
    trainer = GymTrainer(env, agent)
    trainer.load_run_history(history_file_path)

    ############### Define Run inputs and Run ###############
    total_steps = 20000
    render_interval = 1 # 0 is never
    frame_delay = 0.01
    trainer.run_multiple_episodes(total_steps, render_interval, frame_delay, history_file_path=history_file_path, save=False) # multiple runs for up to total_steps

    # ############### Save to file and plot progress ###############
    trainer.save_run_history(history_file_path)
    PlottingTools.plot_smooth(trainer.rewards, silent=False)
