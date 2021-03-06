# TODO: Just learned this is actually called a non-parametric model...apparently I created something very close
# to what the literature calls a replay buffer. Kind of cool! Anyways, might adjust a few naming conventions, since
# Deterministic isn't really the key identifier for this model. There was a note about this in the class homework:
# Course4, Assignment 2, Section 3...
# TODO: Fix terminal transitions. I treat terminal transitions incorrectly, just like any other transition, because
# my model literally doesn't capture the terminal/done field right now. And for that matter, I don't actually record
# terminal transitions...so at least it's consistent. Still, room for improvement.

import random
import numpy as np
import time
import uuid

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
        # This list could be generated as transition_record.keys(), but that creates a painter's problem
        # This costs a bit of memory, but it's much faster.
        self.transition_record_list = []

    def _get_hash(self, state, action, uid=None):
        """
        Use to get unique hashkey. Technically, only the uid is needed, and a streamlined/memory efficient
        implementation would remove state and action. However, for learning purposes, it is conceptual consistent
        with the environment step function, p(s',r|s,a), where s = state, a = action, and p',r can be retrieved using
        self.simulate_env_step()
        :param state: state of model.
        :param action: action selected for this environment update.
        :param uid: unique id. Used to avoid hash collisions.
            None is used to generate a new hash.
            In order to lookup a recorded hash, pass in a valid uid.
        :return:
        """
        if uid is None: # Create new hash
            hash = (tuple(state), action, uuid.uuid4())
        else: # format hash for lookup
            hash = (tuple(state), action, uid)
        return hash

    def append_transition(self, previous_state, previous_action, reward, state):
        hash_key = self._get_hash(previous_state, previous_action)
        self.transition_record_list.append(hash_key)
        self.transition_record[hash_key] = (reward, state)

        if(len(self.transition_record) > self.max_record_count):
            # pop the first element out of the list, and remove the corresponding dictionary entry
            deleted = self.transition_record_list.pop(0)
            del self.transition_record[deleted]

    def sample_transition(self):
        """
        Randomly select a state-action pair.
        """
        state, action, unique_id = random.choice(self.transition_record_list)
        return np.array(state), action, unique_id

    def _simulate_env_step(self, previous_state, previous_action, unique_id):
        hash_key = self._get_hash(previous_state, previous_action, unique_id)
        return self.transition_record[hash_key]

    def simulate(self, update_q):
        # learn from simulation
        for _ in range(self.simulation_frequency):
            # randomly select simulated state and action
            sim_previous_state, sim_previous_action, unique_id = self.sample_transition()
            sim_reward, sim_state = self._simulate_env_step(sim_previous_state, sim_previous_action, unique_id)
            sim_next_action = None # not needed for update under Q-learning algorithm
            # Update q based on simulated values
            update_q(sim_previous_state, sim_previous_action, sim_reward, sim_state, sim_next_action)
