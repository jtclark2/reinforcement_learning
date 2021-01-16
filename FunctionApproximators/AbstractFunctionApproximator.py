import numpy as np

class AbstractStateActionValueApproximator:

    def __init__(self):
        """
        Initialize the weights in a format consistent 
        """
        self.weights = None

    def get_value(self, state, action=None):
        """
        Approximate the state-action_value.

        Inputs:
            :param state: State inputs (this is the raw value that comes out of the env).
            :param action: Action taken in this state to get to state(t+1).

            :return: Approximate value of the State-Action Value function, for the given state and action.
        """
        raise NotImplementedError()

    def get_gradient(self, state, action):
        """
        Find the gradient of the function, where applicable.
            - For tabular representations (Direct representation of one-hot states, Grouping, Coarse Coding, Tiling), this will
        simplify to 1 (or a sparsely represented vector with 1's in all the right places.
            - For Neural networks, tis is the true gradient. Use the backpropagation toolkits to fill this out.

        :param state: State inputs (this is the raw value that comes out of the env).
        :param action: Action taken in this state to get to state(t+1).
        :return:
        """
        raise NotImplementedError


    def _encode_feature_inputs(self, state):
        """
        Encode the state into relevant features. This may require domain specific knowledge - need to work with it a bit
        more to fully understand what this is for various algorithms.
            eg: For tiling, this is a one-hot encoding of the activated tiles.
            eg: For neural networks, this is not needed.
            eg: For any encoding, we could manually engineer features here...such as using sqrt(x^2 +y^2) in a problem
                we know would benefit from a cartesian distance.

        :param state: State inputs (this is the raw value that comes out of the env).
        :param action: Action taken in this state to get to state(t+1).
        :return: Encoding that is consistent with the required input of this state_action function.
        """

    def get_all_action_values(self, state):
        """
        Helper method that uses get_value for create an array of values over all actions in cases of discrete/tabular
        action-spaces.

        :param state: State inputs (this is the raw value that comes out of the env).
        :return: (np.array) of action values.
        """
        action_values = [self.get_value(state, action) for action in range(self.num_actions)]  # forward propagate
        return np.array(action_values)