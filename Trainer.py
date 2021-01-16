import time
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

class Trainer():
    """
    This class works with Gym environment to train agents.
    """
    def __init__(self, env, agent):
        self.rewards = []
        self.env = env
        self.observation = env.reset()
        self.agent = agent

    def train_fixed_steps(self, total_episodes, max_steps_per_run, render_interval=0):
        """
        This may need to rethinking with regard to termination condition...we may want more flexibility than a fixed
        number of steps in more advanced situations.

        :param total_steps: Total planned steps to be taken. Run will be allowed to complete if it starts before
            reaching this value.
        :param max_steps_per_run: Single will be forcefully ended after taking this many steps
        :param render_interval: Render will occur once every render_interval episodes

        :return: None
        """
        episode_count = 0
        greatest_reward = float("-inf")
        render = render_interval
        while(episode_count < total_episodes):
            steps, reward = self.run_env(max_steps_per_run, render)
            self.rewards.append(reward)
            greatest_reward = max(greatest_reward, reward)
            episode_count += 1
            print("Episode: %d/%d, Reward: %f, Best: %f" % (episode_count, total_episodes, reward, greatest_reward))

            # Render every n episodes
            if render_interval != 0 and episode_count % render_interval == 0:
                print(f"Render ON. Attempt: {episode_count} Greatest Reward so far: {greatest_reward}")
                render = True
            else:
                # print(f"Render OFF. Attempt: {episode_count} Greatest Reward so far: {greatest_reward}")
                render = False

    def run_env(self, max_steps, render):
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
        total_reward = 0
        action = self.agent.start(self.observation)
        positions = []
        for step in range(max_steps):
            if render:
                time.sleep(.01)
                self.env.render()
            self.observation, reward, done, info = self.env.step(action)  # take a random action
            pos, vel = self.observation
            positions.append(pos)
            action = self.agent.step(reward, self.observation)
            total_reward += reward

            if (done is True):
                # total_reward -= reward # undo the last rewards (it's wrong now)
                # reward = max(positions)
                # total_reward -= reward # add in the new reward
                self.agent.end(reward)
                break
        step +=1

        self.env.close()
        return step, total_reward

    def save_run_history(self, save_path=""):
        try:
            pickle.dump(self.rewards, open(save_path, "wb"))
            print("Saved training history to: ", save_path)
        except:
            print("Unable to save run history to: ", save_path)

    def load_run_history(self, load_path=""):
        if os.path.isfile(load_path):
            self.rewards = pickle.load(open(load_path, "rb"))
            print("Loaded run history. Current step length: ", len(self.rewards) )
        else:
            print("Warning: Unable to load training_history. Program will proceed without loading.")
            time.sleep(2)

def plot(agent, results, smoothing = 100):
    smoothing = min(len(results)//2, smoothing)
    running_avg = [np.average(results[x:x+smoothing]) for x, _ in enumerate(results[:-smoothing])]
    x_axis = np.array([x for x, _ in enumerate(results)])
    plt.figure(figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(running_avg)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward per Episode")
    plt.yscale("linear")
    # plt.ylim(-200, -75)
    plt.show()


if __name__ == "__main__":

    class AgentStub:
        def __init__(self):
            self.vel = 0
            self.pos = 0
            self.last_action = 0

        def start(self, state):
            action = 0
            self.last_action = action
            return action

        def step(self, reward, state):
            self.pos, self.vel = state
            if self.vel >= 0:
                action = 2
            else:
                action = 0

            self.last_action = action
            return action

        def end(self, reward):
            pass

    # agent = AgentStub()

    from Agents import SarsaAgent, QLearningAgent, SimpleNNAgent, Optimizers
    from Encoders import TileCoder
    import numpy as np

    # ---Specific to MountainCar-v0---
    # # pos, vel
    # env_name = 'MountainCar-v0'
    # state_limits = np.array([[-1.2, 0.5], [-0.07, 0.07]])
    # # Descent results from [16,16] to [32,32]. [64,64] gets unstable (even running 10x longer...not 100% sure why)
    # feature_resolution = np.array([32,32])
    # num_grids = 32
    # # Turns out some epsilon is needed, despite optimistic initialize (run with 0 until plateau, then turn it on to see!)
    # epsilon = 0.0 # 0.01
    # gamma = 1  # discount factor
    # alpha = 1/(2**1) # learning rate: .1 to .5 Converges in a few ~1000 episodes down to about -100

    # # Specific to CartPole-v1
    # # ??? pos, vel, ang_pos, ang_vel ???
    # state_limits = np.array([[-2.5, 2.5], [-2.5, 2.5], [-0.3, 0.3], [-1, 1]])
    # feature_resolution = np.array([16,16,16,16])
    # num_grids = 32
    # epsilon = 0.01  # In case of epsilon greedy exploration
    # gamma = 1  # discount factor
    # alpha = 0.1 # learning rate




    # ############### Environment Setup (and configuration of agent for env) ###############
    # env_name = 'MountainCar-v0'
    # env = gym.make(env_name)
    #
    # ############### Trainer Setup (load run history) ###############
    # trainer = Trainer(env, agent)
    # if(load_status):
    #     trainer.load_run_history(env_name)
    #
    # ############### Define Run inputs and Run ###############
    # total_episodes = 200
    # max_steps = 1000
    # render_interval = 1000 # 0 is never
    # trainer.train_fixed_steps(total_episodes, max_steps, render_interval) # multiple runs for up to total_steps
    #
    # ############### Save to file and plot progress ###############
    # agent.save_learning(env_name)
    # trainer.save_run_history(env_name)
    # plot(agent, np.array(trainer.rewards) )
