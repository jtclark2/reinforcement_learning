import time
from datetime import datetime
import pickle
import os
import numpy as np
from ToolKit.PlottingTools import PlottingTools

# TODO: Pass in a reward shaping function...I know it's frowned upon, beause we all want fully generalized AI,
# but it is a practical tool that should not be overlooked
# TODO: State Shaping. Similar to reward shaping, there are states in which expert knowledge can simplify the problem
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
        self.smoothing = 10

    def run_multiple_episodes(self,
                              target_steps,
                              render_interval,
                              frame_delay=0.01,
                              save_info={"save": False},
                              live_plot=False):
        """
        This may need to rethinking with regard to termination condition...we may want more flexibility than a fixed
        number of steps in more advanced situations.

        :param target_steps: Total planned steps to be taken. Run will be allowed to complete if it starts before
            reaching this value.
        :param render_interval: Render will occur once every render_interval episodes

        :return: None
        """
        step_count = 0
        if len(self.rewards) == 0:
            greatest_reward = float("-inf")
        else:
            greatest_reward = max(self.rewards)
        start_time = datetime.now()
        previous_update_time = 0
        while(step_count < target_steps):
            if render_interval != 0 and self.episode_count % render_interval == 0: # Render every n episodes
                # print(f"Render ON. Attempt: {self.episode_count} Greatest Reward so far: {greatest_reward}")
                render = True
            else:
                render = False

            self.episode_count += 1
            step_increment, reward = self.run_episode(render, frame_delay)
            self.rewards.append(reward)
            greatest_reward = max(greatest_reward, reward)
            step_count += step_increment

            if time.time() - previous_update_time > 10:
                smoothing = min(self.smoothing, len(self.rewards)//2)
                previous_update_time = time.time()
                # TODO: This is a mess now - use the same averaging approach as ADAM (1/(1-b))
                if len(self.rewards) > 1:
                    average_reward = np.average(self.rewards[len(self.rewards)-smoothing:])
                else:
                    average_reward = reward
                if live_plot:
                    PlottingTools.plot_smooth(self.rewards, smoothing=smoothing, silent=True)
                if save_info["save"]:
                    self.save_run_history(save_info["training_history_path"])
                    self.agent.save(save_info["agent_memory_path"], print_confirmation=False)
                current_time = datetime.now()
                time_per_step = (current_time-start_time)/step_count
                steps_remaining = target_steps-step_count
                time_remaining = steps_remaining*time_per_step
                print_friendly_time_remaining = str(time_remaining).split('.', 2)[0] # just shaving off ms noise
                # print("Episode: %d, Steps: %d/%d, Reward: %f, Best: %f, Average: %f" % (self.episode_count, step_count, target_steps, reward, greatest_reward, average_reward))
                # print(f"Current Time: {current_time} | Projected Completion Time: {current_time + time_remaining} | Time remaining: {time_remaining} seconds")
                print(f"Episode: {self.episode_count} | Steps: {step_count}/{target_steps} | "
                      f"Ave Reward: {average_reward} | Time Remaining: {print_friendly_time_remaining}")

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

    def save_run_history(self, save_path="", print_confirmation=False):
        try:
            pickle.dump(self.rewards, open(save_path, "wb"))
            if print_confirmation:
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
    from Agents import HumanAgent, SingleActionAgent, NNAgent
    from ToolKit.PlottingTools import PlottingTools

    ############### Environment Setup (and configuration of agent for env) ###############
    env_name = 'LunarLander-v2' # 'MountainCar-v0', 'LunarLander-v2', 'Breakout-v0', 'Breakout-ram-v0', etc.
    history_file_path = os.getcwd() + "/TrainingHistory/" + env_name
    env = gym.make(env_name)

    ############### Instantiate and Configure Agent ###############
    agent = SingleActionAgent.SingleActionAgent(action=0) # Note that inert agent always sends 0, which is inert in some environments,

    ############### Trainer Setup (load run history) ###############
    trainer = GymTrainer(env, agent)
    trainer.load_run_history(history_file_path)

    ############### Define Run inputs and Run ###############
    save_info = {"save": True,
                 "training_history_path": history_file_path,
                 "agent_memory_path": None}

    trainer.run_multiple_episodes(target_steps=500,
                                  render_interval=1,
                                  frame_delay=0.01,
                                  save_info=save_info,
                                  live_plot=False)

    # ############### Save to file and plot progress ###############
    trainer.save_run_history(history_file_path) # Included for testing (but it's going to be a boring history)
    PlottingTools.plot_smooth(trainer.rewards, silent=False)
