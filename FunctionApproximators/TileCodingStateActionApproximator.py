from FunctionApproximators import AbstractFunctionApproximator
from ToolKit import Tiles3 as tc
import numpy as np

class TileCodingStateActionApproximator:

    def __init__(self, env_name, state_boundaries, num_actions, tile_resolution, num_tilings=8):
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
        self._weights = np.ones((self.num_actions, self._total_possible_tiles)) * 0.0 # TODO: pass in initial values

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

    def get_gradient(self, state = None, action = None):
        """
        Strictly speaking this should be a vector of 1's, np.ones((features))
        Tiling effectively turns the forward function into: [w1,w2,w3,...wn]*[f1,f2,f3,...,fn]
          Where W = [w1,w2,w3,...wn] is the weight vector, and
          F = [f1,f2,f3,...,fn] is the feature vector (1 if tile is active, 0 if tile is not active), and
          the multiplication is elementwise.
        Therefore, given state F, the gradient dW/dv = d [w1,w2,w3,...wn] / dv = [1,1,1,...,1],
        or a vector of 1's of length n.
        However, this representation would have lots of sparse vectors, so this class manages all the multiplication
        with the lookup in the IHT class. Therefore, we don't use the sparse vector. Instead we just use the dense
        representation, an array of 1's representing the dense vector of tiles. Granted, this is absurdly simple in
        this context, because the indexes are inferred, so it's literally always returning the same thing.
        Even the length does not change, as y design, tiles always have the same number active (1 from each tiling),
        so this is np.ones((self._num_tilings))

        :param state: Only included for consistency with interface.
        :param action: Only included for consistency with interface.
        :return: Gradient (which is just 1)
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
        activation = self._get_active_tiles(state)
        self._weights[action][activation] = self._weights[action][activation] + delta / self._num_tilings


if __name__ == "__main__":
    pass