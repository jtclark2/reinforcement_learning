import time
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

class Trainer():
    """
    This class works with Gym environment to train agents.
    """
    def __init__(self, env, agent):
        self.rewards = []
        self.env = env
        self.observation = env.reset()
        self.agent = agent

    def train_fixed_steps(self, total_episodes, max_steps_per_run, render_interval=0, frame_delay=0.05):
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
        average_reward = None
        render = render_interval
        while(episode_count < total_episodes):
            steps, reward = self.run_env(max_steps_per_run, render, frame_delay)
            self.rewards.append(reward)
            greatest_reward = max(greatest_reward, reward)
            learning_rate = 1/100
            if average_reward is None: average_reward = reward
            average_reward = average_reward*(1-learning_rate) + reward*(learning_rate)
            episode_count += 1
            print("Episode: %d/%d, Reward: %f, Best: %f, Average: %f" % (episode_count, total_episodes, reward, greatest_reward, average_reward))

            # Render every n episodes
            if render_interval != 0 and episode_count % render_interval == 0:
                print(f"Render ON. Attempt: {episode_count} Greatest Reward so far: {greatest_reward}")
                render = True
            else:
                # print(f"Render OFF. Attempt: {episode_count} Greatest Reward so far: {greatest_reward}")
                render = False

            # if episode_count % 1000 == 0:
            #     self.agent.save_agent_memory(agent_file_path)
            #     self.save_run_history(history_file_path)

    def run_env(self, max_steps, render, frame_delay):
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
        for step in range(max_steps):
            if render:
                time.sleep(frame_delay)
                self.env.render()
            self.observation, reward, done, info = self.env.step(action)  # take a random action
            # print("Observation: ", self.observationa)
            action = self.agent.step(reward, self.observation)
            total_reward += reward

            if (done is True):
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

    def plot_value_function(self, resolution=50, type = "action_value"):

        min1 = self.agent.value_approximator._state_boundaries[0][0]
        max1 = self.agent.value_approximator._state_boundaries[0][1]
        step_size_1 = (max1-min1)/resolution
        min2 = self.agent.value_approximator._state_boundaries[1][0]
        max2 = self.agent.value_approximator._state_boundaries[1][1]
        step_size_2 = (max2-min2)/resolution

        variable1 = np.zeros((resolution, resolution))
        variable2 = np.zeros((resolution, resolution))
        value = np.zeros((resolution, resolution))

        for index1 in range(resolution):
            var1 = (index1*step_size_1) + min1
            for index2 in range(resolution):
                var2 = (index2*step_size_2) + min2

                variable1[index1, index2] = var1
                variable2[index1, index2] = var2
                value[index1,index2] = np.average(self.agent.value_approximator.get_values(np.array([var1, var2])))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(variable1, variable2, value)
        plt.show()



def plot(agent, results, smoothing = 100):
    smoothing = min(len(results)//2, smoothing)
    running_avg = [np.average(results[x:x+smoothing]) for x, _ in enumerate(results[:-smoothing])]
    x_axis = np.array([x for x, _ in enumerate(results)])
    plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(running_avg)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward per Episode")
    plt.yscale("linear")
    plt.show()


if __name__ == "__main__":
    import gym
    from Agents import HumanAgent

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
            # self.pos, self.vel = state
            # if self.vel >= 0:
            #     action = 2
            # else:
            #     action = 0

            action = 0
            self.last_action = action
            return action

        def end(self, reward):
            pass

        def save_agent_memory(self, save_path):
            pass

        def load_agent_memory(self, load_path):
            return True


    ############### Environment Setup (and configuration of agent for env) ###############
    env_name = 'MountainCar-v0'

    env_name = 'Breakout-v0'
    env_name = 'Breakout-ram-v0'
    history_file_path = os.getcwd() + "/TrainingHistory/" + env_name
    env = gym.make(env_name)

    ############### Instantiate and Configure Agent ###############
    # agent = AgentStub()
    agent = HumanAgent.HumanAgent({"env":env})
    agent_file_path = "" # We're not really saving, so it does not matter
    load_status = agent.load_agent_memory(agent_file_path)

    ############### Trainer Setup (load run history) ###############
    trainer = Trainer(env, agent)
    if(load_status):
        trainer.load_run_history(history_file_path)

    ############### Define Run inputs and Run ###############
    total_episodes = 200
    max_steps = 1000 # turns out most gym.env environments auto-stop (really early in fact)
    render_interval = 1 # 0 is never
    frame_delay = 0.2
    trainer.train_fixed_steps(total_episodes, max_steps, render_interval, frame_delay) # multiple runs for up to total_steps

    # ############### Save to file and plot progress ###############
    agent.save_agent_memory(agent_file_path)
    trainer.save_run_history(history_file_path)
    plot(agent, trainer.rewards)
