import random
import numpy as np

class DeterministicModel:
    """
    This model enables Dyna-Q, allowing learning from replaying actions based on new information.

    Dyna-Q+ is not used, because we are working with state space directly. That means that
    the space is generally continuous, and therefore contains inifinite states. Dyna+ is based on revisitation in
    discrete space, and encentivations visits to areas that have not been recently visited. There are workarounds,
    such as tiling; however they are generally limited and don't extend to higher dimensions very well. I know there are
    density models which try to fix this, though I haven't gotten through the literature yet:
    https://arxiv.org/abs/1606.01868

    max_count_record: Can be used to decrease limit memory, so that the agent can sample in a more relevant way,
    and keep the memory from exploding too much. This would be particularly important with larger observation spaces.
    """

    def __init__(self, simulation_frequency=10, max_record_count=10000):
        # Alternately this could be done based on a timer, or on a separate thread in between real env updates
        self.max_record_count = max_record_count
        self.simulation_frequency = simulation_frequency
        self.transition_record = {}
        # This list could be generated as transition_record.keys(), but that would create a painter's problem
        self.transition_record_list = []

    def _get_hash(self, a, b):
        return (tuple(a), b)

    def record_transition(self, previous_state, previous_action, reward, state):
        if(len(self.transition_record) > self.max_record_count):
            # pop the first element out of the list, and remove the corresponding dictionary entry
            deleted = self.transition_record_list.pop(0)
            try:
                del self.transition_record[deleted]
            except:
                # Whoops, we probably already deleted it, because the hash I've chosen does not guarantee uniqueness
                # Luckily, this doesn't really matter. The likelihood of these collisions is low,
                # and we only lose 1 recorded sample.
                pass

        hash_key = self._get_hash(previous_state, previous_action)
        self.transition_record_list.append(hash_key)
        self.transition_record[hash_key] = (reward, state)

    def select_state_action(self):
        """
        Randomly select a state-action pair.
        """
        state, action = random.choice(self.transition_record_list)
        return np.array(state), action

    def simulate_env_step(self, previous_state, previous_action):
        hash_key = self._get_hash(previous_state, previous_action)
        return self.transition_record[hash_key]

    def simulate(self, update_q):
        # learn from simulation
        for _ in range(self.simulation_frequency):
            # randomly select simulated state and action
            sim_previous_state, sim_previous_action = self.select_state_action()
            sim_reward, sim_state = self.simulate_env_step(sim_previous_state, sim_previous_action)
            sim_next_action = None # not need for update under Q algorithm
            # Update q based on simulated values
            update_q(sim_previous_state, sim_previous_action, sim_reward, sim_state, sim_next_action)
