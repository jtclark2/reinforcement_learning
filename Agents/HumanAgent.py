import keyboard # https://pypi.org/project/keyboard/
import numpy as np


class HumanAgent:
    """
    Just an interface to the keyboard, so the human can be an agent too.
    Responsiveness is poor - though initial tests indicate that is due to graphics lag, rather than control lag.
    Either way, this was not a curiosity. I don't intend to invest much more time in cleaning this one up.
    """
    def __init__(self, agent_info={}):
        self.name = "Human"
        self._format_key_map(agent_info["key_map"]) # Many (not all) gym environments provide env.get_keys_to_action()
        # self._print_controls(env)

    def _print_controls(self, env):
        if 'get_action_meanings' in dir(env):
            action_meanings = env.get_action_meanings()
            i=0
            for key, action in self.key_mapping.items():
                print(f"{action_meanings[i]}: '{key}', ({action})")
                i+=1
            print("If the key is ' ', it may just be spacebar")
        else:
            print("Control mappings not provided. I'd guess arrows or WASD for movement, and space bar to fire.")
            action_meanings = ["Unknown"]* len(env.get_keys_to_action())
    #
    def _format_key_map(self, key_map):
        """
        Reformat the env.get_keys_to_action return into a more convenient format.
        {tuple(gym_key_values), action} --> ordered list of [tuple(keyboard_key_values), action]
            1) convert dict --> list (because dictionaries don't play well with sorting)
            2) sort list
            3) converting ord(ascii) values to keyboard canonical_names.
                # https://github.com/boppreh/keyboard/blob/master/keyboard/_canonical_names.py
        :param env: gym environment
        :side effects: Sets self.key_mapping to the key_mapping (same as return value)
        :return: Dictionary of: {keyboard.keyname, env.action}
        """

        keys_action_pair_list = [(k,v) for k,v in key_map.items()]
        sorted_list_gym_keys = sorted(keys_action_pair_list, key=lambda x:len(x[0]), reverse=True)

        gym_to_keyboard = {276: "left",
                           275: "right",
                           ord(' '): 'space'}

        self.keyboard_action_pairs = []
        for keys, action in sorted_list_gym_keys:
            converted_keys = []
            for i, gym_key in enumerate(keys):
                if gym_key in gym_to_keyboard:
                    keyboard_key = gym_to_keyboard[gym_key]
                else:
                    keyboard_key = chr(gym_key) # most mappings match...we're just adding the keys that are not compatible
                converted_keys.append(keyboard_key)
            self.keyboard_action_pairs.append((converted_keys, action))

        try:
            action_meanings = env.get_action_meanings()
            i = 0
            for keys, action in key_map.items():
                print(f"'{[chr(key) for key in keys]}': {action_meanings[i]}, action=({action})")
                i += 1
            print("If the key is ' ', it may just be spacebar")
        except:
            print("Control mappings not provided.")

    def reset(self, agent_info={}):
        pass

    def select_action(self, state):
        """

        :param state: Ignored (just there to comply with Agent interface.
        :return: action (int), corresponding to the action expected by the env
        """

        for keyboard_keys, action in self.keyboard_action_pairs:
            if np.all([keyboard.is_pressed(key) for key in keyboard_keys]):
                if action != 0: print(action)
                return action

    def start(self, state):
        """
        Kick off a listening thread for keyboard input. Different from previous tries, because I want to capture
        input between frames.
        :param state:
        :return:
        """
        action = self.select_action(state)
        return action

    def step(self, reward, state):
        """
        Advance 1 step in the world, using SARSA update.

        :param reward: Reward value for action taken
        :param state: Current state
        :return: None
        """
        #keyboard input...
        action = self.select_action(state)
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

    def save(self, save_path, print_confirmation=False):
        pass

    def load(self, load_path):
        return True

if __name__ == "__main__":
    from Trainers import GymTrainer
    import gym


    ############### Environment Setup (and configuration of agent for env) ###############
    # Set the env_name to whichever game you want to play! You may need to generate mappings for other environments
    # env_name = 'MountainCar-v0'
    env_name = 'LunarLander-v2' # my top score is 288. Try to beat it!
    env = gym.make(env_name)

    if env_name == 'LunarLander-v2':
        agent_info = {"key_map":{(): 0, (276,): 1, (275,): 3, (275, 276): 0, (ord(' '),): 2}}
    else:
        agent_info = {"key_map": env.get_keys_to_action()}

    agent = HumanAgent(agent_info)

    ############### Create And Configure Agent ###############
    agent = HumanAgent(agent_info)
    agent_file_path = None
    agent.load(agent_file_path)

    ############### Trainer Setup (load run history) ###############
    trainer = GymTrainer.GymTrainer(env, agent)
    trainer.run_multiple_episodes(target_steps=1, render_interval=1) # multiple runs for up to total_steps
