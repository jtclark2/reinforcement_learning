from Encoders import Tiles3 as tc
import numpy as np

class TileCoder:
    def __init__(self, state_boundaries, tile_resolution=8, num_tilings=8):
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
        num_tilings -- int, the number of tilings/grids, where each tiling is another overlapping sheet/grid of tiles
        """

        # no duck typing today
        assert type(state_boundaries) == type(np.array([]))
        assert type(tile_resolution) == type(np.array([]))
        assert type(num_tilings) == type(0)
        assert(state_boundaries.shape[0] == tile_resolution.shape[0])
        assert np.all(state_boundaries.shape[1] == 2)

        self.dimensions = state_boundaries.shape[0] # 0: dims, 1: lower,upper
        self.state_boundaries = state_boundaries
        self.tile_resolution = tile_resolution
        self.num_tilings = num_tilings

        self.memory_allocation = num_tilings
        for res in tile_resolution:
            self.memory_allocation *= res

        self.iht = tc.IHT(self.memory_allocation)

    def get_activations(self, state):
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

        tiles = tc.tiles(self.iht, self.num_tilings, scaled_state)
        return np.array(tiles)




if __name__ == "__main__":
    state_limits = np.array([ [-1.2, 0.5], [-0.07, 0.07]])
    feature_resolution = np.array([8,8])
    num_tilings = 8
    tile_coder = TileCoder(state_limits, feature_resolution, num_tilings)
    tiles = tile_coder.get_activations(np.array([-1.2, -.07]))
    assert(np.all(tiles == [0,1,2,3,4,5,6,7]))

    tiles = tile_coder.get_activations(np.array([0.5, 0.07]))
    assert(np.all(tiles == [8,9,10,11,12,13,14,15]))

    tiles = tile_coder.get_activations(np.array([-1.2, -0.06]))
    assert(np.all(tiles == [ 0, 1, 16, 3, 17, 18, 6, 19]))

    tiles = tile_coder.get_activations(np.array([-1.2, -.07]))
    assert(np.all(tiles == [0,1,2,3,4,5,6,7]))

    print("Completed TileCoder.py - all tests pass.")