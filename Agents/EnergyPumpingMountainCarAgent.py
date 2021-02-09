import numpy as np

class EnergyPumpingMountainCarAgent:
    """
    This agent is specific to mountain car, and does not learn. It is available for testing, benchmark, and
    off-policy learning experiments.
    """

    def __init__(self):
        self.name = "EnergyPumpingMountainCar"
        self.vel = 0
        self.pos = 0
        self.last_action = 0
        self.num_actions = 3

    def reset(self):
        self.vel = 0
        self.pos = 0
        self.last_action = 0

    def select_action(self, state):
        _, vel = state
        return 2 if vel >= 0 else 0

    def start(self, state):
        self.last_action = self.select_action(state)
        return self.last_action

    def step(self, reward, state):
        self.last_action = self.select_action(state)
        return self.last_action

    def end(self, reward):
        pass

    def save_agent_memory(self, save_path):
        pass

    def load_agent_memory(self, load_path):
        """
        Nothing to load.
        :param load_path: Not used.
        :return: True
        """
        return True


if __name__ == "__main__":
    """
    Quick test of the energy pumping agent. Run this file to render an episode.
    """
    import gym
    from Trainers.GymTrainer import GymTrainer

    ############### Instantiate and Configure Agent ###############
    agent = EnergyPumpingMountainCarAgent()

    ############### Environment Setup (and configuration of agent for env) ###############
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    ############### Trainer Setup (load run history) ###############
    trainer = GymTrainer(env, agent)

    ############### Define Run inputs and Run ###############
    total_steps = 1 # will complete the episode that it starts on this step
    render_interval = 1 # 1 is always, 0 is never
    frame_delay = 0.01
    save_info = {"save": False}
    trainer.run_multiple_episodes(total_steps, render_interval, frame_delay=frame_delay) # multiple runs for up to total_steps
