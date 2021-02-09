"""
This is based on homework from Coursera's RL specialization: Course4, assignment2. I'm rewriting it to gain deeper
understanding and integrate into the framework I've built.
"""

import numpy as np
import pickle
from Optimizers.AdamOptimizer import AdamOptimizer
from Models.ReplayBufferModel import ReplayBuffer
from ToolKit.Softmax import softmax
from FunctionApproximators.SingleLayerNNApproximator import SingleLayerNNApproximator

from copy import deepcopy
import os

# TODO: There's no reason the NNAgent needs to be separate from SemiGradientTdControlAgent.
"""
The open question is how much time/effort it's worth to continue pursuing these. The completionist in me wants to,
but as I progress, it really makes more sense to start using mainstream frameworks, and serious incompatibilities 
will arise when I make that jump, deprecating much of this. More importantly, this library was built for learning,
and I've learned what I needed now. Still, I'll list potential improvements here, in case I find the time.
# Differences/Incompatibilities at the moment:
    - This agent does not have a configurable model. It has batch replay hardcoded, whereas semiGrad has a simpler,
      non-batch/non-vectorized replay. The vectorized behavior was required to speed up the NN, and this will probably 
      be the class that I extend/update in the future.
    - This class directly saves private members of the approximator. The approximator should encapsulate this behavior,
      and obviously there is no general expectation that private member variables be consistent across all approximators.
    - This class does runs Expected SARSA, and does not support alternate algorithm configurations (Q-learer, SARSA),
      which is already supported in the SemiGradient model
    - I'd like to clean up the body of the step and end methods, and merge some of the redundant behavior (which is 
      also handled more elegantly in the SemiGradientTdControlAgent
"""
class NNAgent: # (BaseAgent):
    def __init__(self, agent_config=None):
        """
        Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the agent.

        Assume agent_config dict contains:
        {
            network_config: dictionary,
            optimizer_config: dictionary,
            replay_buffer_size: integer,
            minibatch_sz: integer,
            num_replay_updates_per_step: float
            discount_factor: float,
        }
        """
        self.name = "NN_expected_sarsa"
        # It is expected that a config be passed in, but I leave these reasonable values for reference.
        # They were tuned for lunar_lander (fairly ad-hoc, b/c I don't have the gpu for richer sweeps)
        if agent_config is None:
            agent_config = {
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
                'minibatch_sz': 32, # originally 8, I increased to get back efficiency of vectorization
                'num_replay_updates_per_step': 4,
                'gamma': 0.99,
                'tau': 0.001, # default at 0.001 (controls exploration)
                'env_name': "UnknownEnv" # for naming of save files
            }

        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'],
                                          agent_config['minibatch_sz'], agent_config.get("seed"))
        self.network = SingleLayerNNApproximator(agent_config['network_config'])
        self.optimizer = AdamOptimizer(self.network.layer_sizes, agent_config["optimizer_config"])
        self.num_actions = agent_config['network_config']['num_actions']
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.discount = agent_config['gamma']
        self.tau = agent_config['tau']

        self.rand_generator = np.random.RandomState(agent_config.get("seed"))

        self.last_state = None
        self.last_action = None

        self.sum_rewards = 0
        self.episode_steps = 0

    def reset(self):
        pass

    def select_action(self, state):
        """
        Args:
            state (Numpy array): the state.
        Returns:
            the action.
        """
        action_values = self.network.get_action_values(state)
        probs_batch = softmax(action_values, self.tau)
        action = self.rand_generator.choice(self.num_actions, p=probs_batch.squeeze())
        return action

    def _get_delta(self, experiences, current_network):
        """
        Purpose:
            Runs the parameter update, to optimize the action-value approximation function,
            based on recorded experiences.
        Args:
            experiences (Numpy array): The batch of experiences including the states, actions,
                                       rewards, terminals, and next_states.
            discount (float): The discount factor.
            network (ActionValueNetwork): The latest state of the network that is getting replay updates.
            current_network (ActionValueNetwork): The fixed network used for computing the targets,
                                            and particularly, the action-values at the next-states.
        """

        # Get states, action, rewards, terminals, and next_states from experiences
        states, actions, rewards, terminals, next_states = map(list, zip(*experiences))
        states = np.concatenate(states) # The batch of states with the shape (batch_size, state_dim).
        next_states = np.concatenate(next_states) # The batch of next states with the shape (batch_size, state_dim).
        rewards = np.array(rewards) # The batch of rewards with the shape (batch_size,).
        terminals = np.array(terminals) # The batch of terminals with the shape (batch_size,).
        batch_size = states.shape[0]

        # Compute action values at the current/previous state for all actions using network
        # I've been calling prev, but literature often calls it current
        q_vec = current_network.get_action_value(states, actions)

        # Compute TD error: shape (batch_size)
        # TODO: If I add Q-learning and Sarsa, this is where I'd put em'
        # Expected SARSA (next Q averaged over policy)
        q_next_mat = current_network.get_action_values(next_states) # action values (batch_size, num_actions)
        # TODO (priority=low): Could pass in the policy (softmax, epsilon greedy, etc.)
        probs_mat = softmax(q_next_mat, self.tau) # Compute policy at next state (batch_size, num_actions)
        q_next_vec = self.discount*np.sum(q_next_mat * probs_mat, 1) * (1 - terminals) # Zero for terminal transitions

        # Compute TD errors for actions taken
        delta_vec = rewards +  q_next_vec - q_vec # shape (batch_size)

        # Batch Indices is an array from 0 to the batch_size - 1.
        batch_indices = np.arange(batch_size)

        # Make a td error matrix of shape (batch_size, num_actions)
        # delta_mat has non-zero value only for actions taken
        delta_mat = np.zeros((batch_size, self.num_actions))
        delta_mat[batch_indices, actions] = delta_vec

        return delta_mat

    def start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = np.array([state])
        self.last_action = self.select_action(self.last_state)
        return self.last_action

    def step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

        self.sum_rewards += reward
        self.episode_steps += 1

        # Make state an array of shape (1, state_dim) to add a batch dimension and
        # to later match the get_action_values() and get_TD_update() functions
        state = np.array([state])

        # Select action
        action = self.select_action(state)

        # Append new experience to replay buffer
        # Note: look at the replay_buffer append function for the order of arguments

        terminal = False
        self.replay_buffer.append_transition(self.last_state, self.last_action, reward, terminal, state)

        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            for _ in range(self.num_replay):
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample_transition()

                delta_mat = self._get_delta(experiences, deepcopy(self.network))
                self.network.upate_weights(delta_mat, self.optimizer, experiences)

        # Update the last state and last action.
        self.last_state = state
        self.last_action = action

        return action

    def end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1

        # Set terminal state to an array of zeros
        state = np.zeros_like(self.last_state)

        # Append new experience to replay buffer
        # Note: look at the replay_buffer append function for the order of arguments

        terminal = True
        self.replay_buffer.append_transition(self.last_state, self.last_action, reward, terminal, state)

        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample_transition()

                # Call optimize_network to update the weights of the network
                self._get_delta(experiences, current_q)

    def message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")

    def save(self, save_path, print_confirmation=True):
        # save: network parameters, replay_buffer?, agent_config
        try:
            save_obj = (self.network, self.replay_buffer)
            pickle.dump(save_obj, open(save_path, "wb"))
            if print_confirmation:
                print("Agent memory saved to: ", save_path)
        except:
            print("Unable to save agent memory to specified directory: %s " % save_path)

    def load(self, load_path):
        try:
            loaded_data = pickle.load(open(load_path, "rb"))
        except:
            print("Unable to find file: %s\n Proceeding with new agent. Memory has not been loaded." % load_path)
            return False

        (network, replay_buffer) = loaded_data
        if self.replay_buffer.check_config(replay_buffer):
            self.replay_buffer.load(replay_buffer)
        else:
            return False

        if self.network.check_config_match(network):
            self.network.load(network)
        else:
            return False

        return True

################ Section 6: Run it! ################
if __name__ == "__main__":
    import gym
    from Agents import HumanAgent, SingleActionAgent, NNAgent
    from Trainers.GymTrainer import GymTrainer

    ############### Configurable Params ###############
    total_steps = 500000
    render_interval = 1 # Use 0 for training, 1 to watch, and large numbers for periodic updates
    load = True
    save = True
    live_plot = False
    frame_delay = 0.005

    ############### Create environment ###############
    # env_name = 'MountainCar-v0' # 'MountainCar-v0', 'Breakout-v0', 'Breakout-ram-v0', etc.
    env_name = 'LunarLander-v2'
    # env_name = 'Breakout-ram-v0'
    env = gym.make(env_name)

    ############### Configure Agent ###############
    # agent = HumanAgent.HumanAgent({"env":env})
    # agent = InertAgent.InertAgent()
    observations = env.reset()
    agent_config = {
        'network_config': {
            'state_dim': len(observations),
            'num_hidden_units': 256,
            'num_actions': env.action_space.n
        },
        'optimizer_config': {
            'step_size': 1e-4,
            'beta_m': 0.9,
            'beta_v': 0.99,
            'epsilon': 1e-8
        },
        'replay_buffer_size': 50000,
        'minibatch_sz': 64,  # originally 8, I increased to improve efficiency of vectorization and stabilize results in sparse reward spaces.
        'num_replay_updates_per_step': 4,
        'gamma': 0.99,
        'tau': 0.001,  # default at 0.001 (controls exploration)
        'env_name': env_name  # for naming of save files
    }
    print(agent_config['network_config'])
    agent = NNAgent.NNAgent(agent_config) # no args --> default params

    ############### Create Trainer ###############
    trainer = GymTrainer(env, agent)

    ############### Load Files ###############
    agent_file_path = os.getcwd() + r"/AgentMemory/Agent_%s_%s.p" % (agent.name, env_name)
    trainer_file_path = os.getcwd() + r"/../Trainers/TrainingHistory/History_%s_%s.p" % (agent.name, env_name)

    if load:
        load_status = agent.load(agent_file_path)
        if (load_status):
            trainer.load_run_history(trainer_file_path)

    ############### Train ###############
    save_info = {"save": True,
                 "training_history_path": trainer_file_path,
                 "agent_memory_path": agent_file_path}

    trainer.run_multiple_episodes(target_steps=total_steps,
                                  render_interval=render_interval,
                                  frame_delay=frame_delay,
                                  save_info=save_info,
                                  live_plot=live_plot)

    # ############### Save to file and plot progress ###############
    if save:
        agent.save(agent_file_path)
        trainer.save_run_history(trainer_file_path)
