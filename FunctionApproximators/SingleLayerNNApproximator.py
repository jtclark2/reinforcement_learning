import numpy as np

class TileCodingStateActionApproximator:

    def __init__(self, network_config):
        """
        Initializes the Neural Network
        """
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        self.num_actions = network_config.get("num_actions")

        self.rand_generator = np.random.RandomState(network_config.get("seed"))

        # Specify self.layer_size which shows the number of nodes in each layer
        ### START CODE HERE (~1 Line)
        self.layer_sizes = [self.state_dim, self.num_hidden_units, self.num_actions]
        ### END CODE HERE

        # Initialize the weights of the neural network
        # self.weights is an array of dictionaries with each dictionary corresponding to
        # the weights from one layer to the next. Each dictionary includes W and b
        self.weights = [dict() for i in range(0, len(self.layer_sizes) - 1)]
        for i in range(0, len(self.layer_sizes) - 1):
            self.weights[i]['W'] = self._init_saxe(self.layer_sizes[i], self.layer_sizes[i + 1])
            self.weights[i]['b'] = np.zeros((1, self.layer_sizes[i + 1]))

    def _init_saxe(self, rows, cols):
        """
        Args:
            rows (int): number of input units for layer.
            cols (int): number of output units for layer.
        Returns:
            NumPy Array consisting of weights for the layer based on the initialization in Saxe et al.
        """
        tensor = self.rand_generator.normal(0, 1, (rows, cols))
        if rows < cols:
            tensor = tensor.T
        tensor, r = np.linalg.qr(tensor)
        d = np.diag(r, 0)
        ph = np.sign(d)
        tensor *= ph

        if rows < cols:
            tensor = tensor.T
        return tensor

    def get_values(self, state):
        """
        Args:
            s (Numpy array): The state.
        Returns:
            The action-values (Numpy array) calculated using the network's weights.
        """

        W0, b0 = self.weights[0]['W'], self.weights[0]['b'] # weights for layer 0 (input layer)
        psi = np.dot(state, W0) + b0 # Linear vector result of layer 0 (not activated yet)
        x = np.maximum(psi, 0) # activation of layer 0

        W1, b1 = self.weights[1]['W'], self.weights[1]['b'] # weights of layer 1
        q_vals = np.dot(x, W1) + b1 # linear combination of layer 1...I guess we're not applying softmax activation? Must be done in a later step

        return q_vals

    def get_value(self, state, action):
        """
        Forward pass of NN.
        :param state:
        :param action:
        :return:
        """
        raise NotImplementedError
        return action_value

    def _get_gradient(self, state = None, action = None):
        return np.ones((self._num_tilings)) # 1

    def check_config_match(self, approximator):
        """
        Does this configuration match an externally provided config?
        :param approximator: The approximator that this config is being compared to
        :return: match: True/False
        """
        # env_name = np.all(self.env_name == approximator.env_name)
        raise NotImplementedError

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