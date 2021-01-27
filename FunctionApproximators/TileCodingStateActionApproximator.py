from FunctionApproximators import AbstractFunctionApproximator
from ToolKit import Tiles3 as tc
import numpy as np

class TileCodingStateActionApproximator:

    def __init__(self, env_name, state_boundaries, num_actions, tile_resolution, num_tilings=8, initial_value = 0.0):
        """
        Initializes the Tile Coder.
        The tile encoder wraps tiles3.iht (index hash table). This enables easier generation of a tile encoding.

        Initializers:
        state_boundaries = np.array of tuples (min,max) in the units of choice...
            For example, if encoding pos, vel:

        activation_size -- int, the size of the index hash table (total memory allocation), typically a power of 2
        tile_resolution -- Technically num_tiles is more accurate, since this is a count, but resolution strongly
            conveys that this divides a grid into tiles
            1D np.array(), in which each element is the size of the feature space in that dimension.
            Alternatively, this can be given as an integer, broadcasting across all dims
        num_grids -- int, the number of tilings/grids, where each tiling is another overlapping sheet/grid of tiles
            - chose num_grids, because num_grids was too similar to num_tiles, causing confusion
        """
        # Data Verification
        assert type(state_boundaries) == type(np.array([]))
        assert type(tile_resolution) == type(np.array([]))
        assert type(num_tilings) == type(0)
        assert(state_boundaries.shape[0] == tile_resolution.shape[0])
        assert np.all(state_boundaries.shape[1] == 2)

        self.env_name = env_name
        self._dimensions = state_boundaries.shape[0] # 0: dims, 1: lower,upper
        self._state_boundaries = state_boundaries
        self._tile_resolution = tile_resolution
        self._num_tilings = num_tilings
        self.num_actions = num_actions

        self._total_possible_tiles = num_tilings # all possible tiles across all tilings
        for res in tile_resolution:
            self._total_possible_tiles *= res
        self._iht = tc.IHT(self._total_possible_tiles)

        # Consider generalizes this in the future. This is optimistic for anything < 1.
        self._weights = np.ones((self.num_actions, self._total_possible_tiles)) * initial_value # TODO: pass in initial values

    def get_values(self, state):
        """
        Although the parent method works fine, this is slightly more optimized. Gets values for all actions.
        :param state:
        :return:
        """
        features = self._get_active_tiles(state)
        action_values = np.sum(self._weights[:, features], axis=1)
        return action_values

    def get_value(self, state, action):
        features = self._get_active_tiles(state)
        action_value = np.sum(self._weights[action, features])
        return action_value

    def _get_gradient(self, state = None, action = None):
        """
        Gets the dense representation of the gradient, corresponding to the [action, state_activations].
        Each of these feature vectors are just
            V_hat = W*X = [w1,w2,w3,...wn]*[x1,x2,x3,...,xn] = [w1*x1, w2*x1, w3*x3...,wn*xn], where:
            W = [w1,w2,w3,...wn] is the weight vector, and
            X = [x1,x2,x3,...,xn] is the feature vector of [action, tile activations] (1 if tile is active and taking
            the corresponding action, 0 for everything else]
            active), and
            Multiplication is element-wise
        Therefore, given feature activation vector F, the gradient:
            dW/dV = d [w1,w2,w3,...wn] / dv = [x1,x2,x3,...,xn] = [0,1,0,...,0]
        In general, for all linear functions, we would just say the gradient is X = [x1,x2,x3,...,xn]. For tile coding,
        the features vectors are just sparse vectors, where if the state is in a tile, x = 1, not in the tile x = 0.
        That leaves a sparse vector of most '0's, with '1's just in the activated locations.

        W has the same dimensions grad = dW/dV has the same dimensions as W, a vector of length =
        self.num_actions * self._total_possible_tiles. However, for convenience, in this implementation, purely for
        convenience, W is represented as matrix [self.num_actions, self._total_possible_tiles]

        If we were using the full representation described just above, then the code would look like this:
            full_grad = np.zeros((action,self._total_possible_tiles))
            activations = self._get_active_tiles(state)
            full_grad[action, activations] = np.ones((self._num_tilings))
            return full_grad

        However, we would waste tons of cpu cycles on operations that are all 0. So we just run the gradient for the
        activated elements. As mentioned, these are all 1's. And in fact we know that the there are self._num_tilings
        total activations, so that is the vector length. Thus, we just return a vector of '1's. This entire class works
        on this assumption, so the calling members know that they are really getting grad=full_grad[action][activation]

        :param state: Only included for consistency with interface.
        :param action: Only included for consistency with interface.
        :return: Dense gradient. grad = full_grad[action][activation]. Read full comment block for details
        """
        return np.ones((self._num_tilings)) # 1

    def _get_active_tiles(self, state):
        """
        Takes in state and encodes as tile indices.

        Arguments:
        state -- The state as an np.array() of length self.dimensions, to be encoded as tiles

        returns:
        tiles - np.array, active tiles
        """
        assert(state.shape[0] == self._dimensions)

        scaled_state = np.zeros(self._dimensions)
        for dim, feature in enumerate(state):
            val_min, val_max = self._state_boundaries[dim]
            val_range = val_max - val_min
            scaled_state[dim] = (feature - val_min) / val_range * self._tile_resolution[dim]

        tiles = tc.tiles(self._iht, self._num_tilings, scaled_state)
        return np.array(tiles)

    def check_config_match(self, approximator):
        """
        Does this configuration match an externally provided config?

        The tilings are randomly generated, so self._iht will never match config._iht. Therefore 2 separate tilings
        can never be combined; however, given the same parameters, the old one can just be overwritten.
        :param config:
        :return:
        """

        env_name = np.all(self.env_name == approximator.env_name)
        bound = np.all(self._state_boundaries == approximator._state_boundaries)
        res = np.all(self._tile_resolution == approximator._tile_resolution)
        num = self._num_tilings == approximator._num_tilings
        return bound and res and num and env_name

    def update_weights(self, delta, state, action):
        """
        Handles updating the weights that estimate the Value function. Z
        :param delta:
        :param state:
        :param action:
        :return:
        """
        activation = self._get_active_tiles(state)
        # The gradient is just a vector of '1's for tile-coders using mse, so it's just wasting cpu cycles; however,
        # this project is about clarity, not efficiency. Also, if we were to generalize to other error functions,
        # this formula would still be correct
        self._weights[action][activation] = self._weights[action][activation] + (delta / self._num_tilings) * self._get_gradient(state, action)


if __name__ == "__main__":
    pass