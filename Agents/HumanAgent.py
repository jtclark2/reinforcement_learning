import keyboard
from enum import Enum


class KeyMap(Enum):
    Left = ('left', 0)
    No_Force = ('no_force', 1)
    Right = ('right', 2)


class HumanAgent:
    """
    Just an interface to the keyboard, so the human can be an agent too.
    """
    def __init__(self, agent_info={}):
        self.name = "Human"

    def reset(self, agent_info={}):
        pass

    def select_action(self, state):
        """

        :param state: Ignored (just there to comply with Agent interface.
        :return: action (int), corresponding to the action expected by the env
        """
        action = KeyMap.No_Force
        if keyboard.is_pressed('left'):  # 'a'
            action = KeyMap.Left
        elif keyboard.is_pressed('right'):  # 'd'
            action = KeyMap.Right
        # print(f"action: {action.name}: {action.value} ")
        return action.value[1]

    def start(self, state):
        """
        Kick off a listening thread for keyboard input. Different from previous tries, because I want to capture
        input between frames.
        :param state:
        :return:
        """
        action = self.select_action()
        return action

    def step(self, reward, state):
        """
        Advance 1 step in the world, using SARSA update.

        :param reward: Reward value for action taken
        :param state: Current state
        :return: None
        """
        #keyboard input...
        action = self.select_action()
        return action

    def end(self, reward):
        """
        Very last step.
        :param reward: Reward gained for action taken
        :return: None
        """
        pass

    def message(self):
        pass

    def save_agent_memory(self, save_path):
        pass

    def load_agent_memory(self, load_path):
        pass

if __name__ == "__main__":
    import Trainer
    import gym

    ############### Environment Setup (and configuration of agent for env) ###############
    # env_name = 'CartPole-v1'
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    ############### Create And Configure Agent ###############
    agent_info = None
    agent = HumanAgent(agent_info)
    agent_file_path = None
    load_status = agent.load_agent_memory(agent_file_path)

    ############### Trainer Setup (load run history) ###############
    trainer_file_path = None
    trainer = Trainer.Trainer(env, agent)
    total_episodes = 1
    max_steps = 1000
    render_interval = 1 # # I better be able to see if I'm playing
    trainer.train_fixed_steps(total_episodes, max_steps, render_interval) # multiple runs for up to total_steps
