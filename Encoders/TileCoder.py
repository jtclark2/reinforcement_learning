from Encoders import Tiles3 as tc
import numpy as np

class TileCoder:
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

        tiles = tc.tiles(self.iht, self.num_grids, scaled_state)
        return np.array(tiles)




if __name__ == "__main__":
    state_limits = np.array([ [-1.2, 0.5], [-0.07, 0.07]])
    feature_resolution = np.array([8,8])
    num_grids = 8
    tile_coder = TileCoder(state_limits, feature_resolution, num_grids)
    tiles = tile_coder.get_activations(np.array([-1.2, -.07]))
    assert(np.all(tiles == [0,1,2,3,4,5,6,7]))

    tiles = tile_coder.get_activations(np.array([0.5, 0.07]))
    assert(np.all(tiles == [8,9,10,11,12,13,14,15]))

    tiles = tile_coder.get_activations(np.array([-1.2, -0.06]))
    assert(np.all(tiles == [ 0, 1, 16, 3, 17, 18, 6, 19]))

    tiles = tile_coder.get_activations(np.array([-1.2, -.07]))
    assert(np.all(tiles == [0,1,2,3,4,5,6,7]))

    print("Completed TileCoder.py - all tests pass.")