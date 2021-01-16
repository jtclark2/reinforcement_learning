"""
See class description.
"""

import numpy as np


class RawStateEncoder:
    """
    This is the encoder class that is used when no encoder is needed, but the Encoder interface is required.
    get_activations just returns the input state.
    """
    @classmethod
    def get_activations(cls, state):
        return state

if __name__ == "__main__":
    state = np.array([1,0,0,0,0,1,2])
    encoder = RawStateEncoder()
    features = encoder.get_activations(state)
    assert np.all(features == state)

    print("RawStateEncoder tests pass.")
