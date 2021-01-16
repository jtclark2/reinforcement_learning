from FunctionApproximators import AbstractFunctionApproximator
from Encoders import Tiles3 as tc
import numpy as np

class TileCodingStateActionApproximator(AbstractFunctionApproximator):

    def __init__(self, state_boundaries, tile_resolution=8, num_grids=8):
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

        # This would generally work with lists, but let's simplify and optimize life
        assert(type(state_boundaries) == type(np.array([])))
        self.dimensions = state_boundaries.shape[0] # 0: dims, 1: lower,upper

        try:
            if(type(tile_resolution) == type(0)):
                num_grids = np.array([num_grids for _ in range(self.dimensions)])
        except:
            raise Exception("Make sure feature_resolution is an np.array() or an integer.")

        self.state_boundaries = state_boundaries
        self.num_activations = num_grids # just a starting point, not finished until after loop
        for res in tile_resolution:
            self.num_activations *= res
        if self.num_activations > 2**21:
            # Chose to raise an exception, rather than just warn... I would definitely ignore my own warnings.
            raise Exception("You really shouldn't be using tiling for such a large problem..." +
                            "I don't care how much you generalize, this won't end well.")
        self.iht = tc.IHT(self.num_activations)
        self.tile_resolution = tile_resolution
        self.num_grids = num_grids
        self.weights = np.ones((self.num_actions, self.encoder.num_activations)) * agent_info.get("initial_weights", 0.0)

    def get_all_action_values(self, state):
        """
        Although the parent method works fine, this is slightly more optimized.
        :param state:
        :return:
        """
        features = self._encode_features(state)
        action_values = np.sum(self.w[:, features], axis=1)
        return

    def get_value(self, state, action=None):
        features = self._encode_features(state)
        action_value = np.sum(self.w[action, features])
        return action_value

    def get_gradient(self, state = None, action = None):
        """
        Just send it back
        :param state: Only included for consistency with interface.
        :param action: Only included for consistency with interface.
        :return: Gradient (which is just 1)
        """
        return 1

    def _encode_features(self, state):
        """
        Takes in state and encodes as tile indices.

        Arguments:
        state -- The state as an np.array() of length self.dimensions, to be encoded as tiles

        returns:
        tiles - np.array, active tiles
        """
        # print("Dims: ", self.dimensions)
        assert(state.shape[0] == self.dimensions)

        scaled_state = np.zeros(self.dimensions)
        for dim, feature in enumerate(state):
            val_min, val_max = self.state_boundaries[dim]
            val_range = val_max - val_min
            scaled_state[dim] = (feature - val_min) / val_range * self.tile_resolution[dim]

        tiles = tc.tiles(self.iht, self.num_grids, scaled_state)
        return np.array(tiles)




if __name__ == "__main__":