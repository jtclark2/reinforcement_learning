import time
import pickle
import os

# TODO: Pass in a reward shaping function...I know it's frowned upon, beause we all want fully generalized AI,
# but it is a practical tool that should not be overlooked
class GymTrainer():
    """
    This class works with Gym environment to train agents.
    """

    def __init__(self, env, agent):
        self.rewards = []
        self.env = env
        self.observation = env.reset()
        self.agent = agent

    def run_multiple_episodes(self, total_episodes, render_interval=0, frame_delay=0.01):
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
            reward = self.run_episode(render, frame_delay)
            self.rewards.append(reward)
            greatest_reward = max(greatest_reward, reward)
            learning_rate = 1/100
            if average_reward is None: average_reward = 0
            average_reward = average_reward*(1-learning_rate) + reward*(learning_rate)
            episode_count += 1
            if episode_count % 10 == 0:
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
        return average_reward

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
        while True:
            if render:
                time.sleep(frame_delay)
                self.env.render()
            self.observation, reward, done, info = self.env.step(action)  # take a random action
            # print("Observation: ", self.observation)
            total_reward += reward

            if (done is True):
                self.agent.end(reward)
                break
            else:
                action = self.agent.step(reward, self.observation)

        self.env.close()
        return total_reward

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




if __name__ == "__main__":
    import gym
    from Agents import HumanAgent
    from ToolKit.PlottingTools import PlottingTools


    # TODO: I want to clean up the setup for specific environments...each algorithm needs different setup depending on
    # problem. The differences are just hyperparameters, which is a reasonable level of adjustment. I just need
    # a slightly more organized/encapsulated way to set them up. See SemiGradientTdAgent example
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
    trainer = GymTrainer(env, agent)
    if(load_status):
        trainer.load_run_history(history_file_path)

    ############### Define Run inputs and Run ###############
    total_episodes = 200
    max_steps = 1000 # turns out most gym.env environments auto-stop (really early in fact)
    render_interval = 1 # 0 is never
    frame_delay = 0.2
    trainer.run_multiple_episodes(total_episodes, render_interval, frame_delay) # multiple runs for up to total_steps

    # ############### Save to file and plot progress ###############
    agent.save_agent_memory(agent_file_path)
    trainer.save_run_history(history_file_path)
    PlottingTools.plot_smooth(trainer.rewards)
