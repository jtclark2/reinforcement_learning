import numpy as np

# TODO: (optional) consider merging with Deterministic model...This has the batching, but that one has a full simulate
# step. Internally this one is simpler. This structure is better for now, though the other one might be fun to extend
# to some more efficient simulation strategies I've been thinking about
class ReplayBuffer:
    """
    This is a non-parametric (direct playback) model equipped for mini-batching. It could probably be merged in with
    the Deterministic model. They're almost identical, but this one has a sample method that returns vetors of
    size=self.minibatch_size.
    """
    def __init__(self, size, minibatch_size, seed):
        """
        Args:
            size (integer): The size of the replay buffer.
            minibatch_size (integer): The sample size.
            seed (integer): The seed for the random number generator.
        """
        self.buffer = []
        self.minibatch_size = minibatch_size
        self.rand_generator = np.random.RandomState(seed)
        self.max_size = size

    def append_transition(self, state, action, reward, terminal, next_state):
        """
        Args:
            state (Numpy array): The state.
            action (integer): The action.
            reward (float): The reward.
            terminal (integer): 1 if the next state is a terminal state and 0 otherwise.
            next_state (Numpy array): The next state.
        """
        if len(self.buffer) == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, reward, terminal, next_state])

    def sample_transition(self):
        """
        Returns:
            A list of transition tuples including state, action, reward, terinal, and next_state
        """
        idxs = self.rand_generator.choice(np.arange(len(self.buffer)), size=self.minibatch_size)
        return [self.buffer[idx] for idx in idxs]

    def size(self):
        return len(self.buffer)

    def check_config(self, loaded):
        assert(loaded.buffer)
        return True

    def load(self, loaded):
        self.buffer = loaded.buffer