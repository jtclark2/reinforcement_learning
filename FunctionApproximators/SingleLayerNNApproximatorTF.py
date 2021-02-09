import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax

class SingleLayerNNApproximatorTF:
    """
    As with this entire library, this is all for the sake of learning. This is effectively the impplementation of
    a 2-layer (state, hidden, actions) fully connected NN.
    In TF, the underlying model would be:
        model = Sequential([
            Dense(num_hidden, activation='relu', input_shape=(state_dim)),
            Dense(num_actions),
            # Softmax() # softmax implemented separately, as a policy, rather than part of the approximator
            ])
    """
    def __init__(self, network_config):
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        self.num_actions = network_config.get("num_actions")

        self.rand_generator = np.random.RandomState(network_config.get("seed")) # Todo: just use RandomState directly?

        # Specify self.layer_size which shows the number of nodes in each layer
        self.layer_sizes = [self.state_dim, self.num_hidden_units, self.num_actions]

        # Initialize model
        self.model = Sequential([
            Dense(self.num_hidden_units ,input_shape = (self.state_dim)),
            Dense(self.num_actions)
        ])

        # TODO: Open question - do epochs exist in an RL agent...I don't think they would...I think we'd just turn
        # the experience replay buffer into a dataset, which would get wrapped as an iterable, which would then get fed into
        # the update method continuously

        # TODO (should work): LR Update
        # How do I do this? LR is going to be TD in this case...
        # Set self.learning_rate, and then use the LearningRateSchedular, on_batch_begin (that might need to be custom)
        # in order to update learning_rate=self.learning_rate*self.td_error, where td_error is of course being set by
        # our agent's update
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)


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

    def _get_gradient_td_update(self, state, delta_mat):
        """
        Gets all the gradients, and then multiplies them all by the TD error, playing the role alpha would play in
        a supervised back propagation.

        Args:
            state (Numpy array): The state, of shape (batch_size,state_size)
            delta_mat (Numpy array): TD-error matrix of shape (batch_size, num_actions). Each row of delta_mat
            correspond to one state in the batch. Each row represents action as a one-hot vector, only one non-zero
            element, which is the TD-error corresponding to the action taken.
        Returns:
            The TD update (Array of dictionaries with gradient times TD errors) for the network's weights
        """

        # td_update has the same structure as self.weights, that is an array of dictionaries.
        # td_update[0]["W"], td_update[0]["b"], td_update[1]["W"], and td_update[1]["b"] have the same shape as
        # self.weights[0]["W"], self.weights[0]["b"], self.weights[1]["W"], and self.weights[1]["b"] respectively
        td_update = [dict() for i in range(len(self._weights))] # len(self.weights should always be 2 in this case)

        W0, b0 = self._weights[0]['W'], self._weights[0]['b']
        W1, b1 = self._weights[1]['W'], self._weights[1]['b']

        psi = np.dot(state, W0) + b0  # forward prop of layer 0
        x = np.maximum(psi, 0)  # activation of layer 0
        dx = (psi > 0).astype(float)  # derivative of relu: [-inf,0) = 0; [0,inf] = 1

        # backprop update into layer 1 (linear only, since there is no activation on this output)
        # Layer1 = Psi = xw0+b0 --> dPsi/dw0 = x, dPsi/db0 = 1; Then apply delta.
        td_update[1]['W'] = np.dot(x.T, delta_mat) * 1. / state.shape[0]
        td_update[1]['b'] = np.sum(delta_mat, axis=0, keepdims=True) * 1. / state.shape[0]

        # backprop updat into layer 0
        delta_mat = np.dot(delta_mat, W1.T) * dx  # backprop of the layer 0 activation, dx
        # Value = V = state*w0+b0 --> dV/dw0 = state, dV/db0 = 1; Then apply delta.
        td_update[0]['W'] = np.dot(state.T, delta_mat) * 1. / state.shape[0]  # backprop into layer 0
        td_update[0]['b'] = np.sum(delta_mat, axis=0, keepdims=True) * 1. / state.shape[0]  # backprop into layer 0

        return td_update

    def _get_weights(self):
        """
        Returns:
            A copy of the current weights of this network.
        """
        return deepcopy(self._weights)

    def _set_weights(self, weights):
        """
        Args:
            weights (list of dictionaries): Consists of weights that this network will set as its own weights.
        """
        self._weights = deepcopy(weights)

    def upate_weights(self, delta_mat, optimizer, experiences):
        # Not sure how expensive this unpacking is...I do it here, and one other time, so there is an opportunity
        # for optimization; however, I prefer this structure because it expresses a clean seperation between the
        # optimizer, the approximator, and the agent.
        states, actions, rewards, terminals, next_states = map(list, zip(*experiences))
        states = np.concatenate(states)

        # Pass delta_mat to compute the TD errors times the gradients of the network's weights from back-propagation
        td_update = self._get_gradient_td_update(states, delta_mat)

        # Pass network.get_weights and the td_update to the optimizer to get updated weights
        weights = self._get_weights()
        weights = optimizer.update_weights(weights, td_update)  # td_errors_times_gradients = td_update
        self._set_weights(weights)

    def get_action_value(self, states, actions):
        """
        Return the action_value corresponding to the input state/action. This is vectorized, so it works on vectors of
        states/actions, but differs from get_action_values because it selects the action, returning a vector of
        action-values of dim (batch_size), rather than a matrix of (batch_size, num_actions)
        :param state: The state
        :param action:
        :return:
        """

        action_value_matrix = self.get_action_values(states)
        batch_indices = np.arange(action_value_matrix.shape[0])  # Batch Indices is an array from 0 to the batch size - 1.
        action_value_vector = action_value_matrix[batch_indices, actions]  # actions shape is (batch_size,)
        return action_value_vector

    def get_action_values(self, state):
        """
        Purpose: This method performs forward propagation, in order to calculate action_values estimations for all
        actions from a given state.
        Note: There is a large opportunity to vectorize this over batches, but let's be honest, I'm switching to Tensorflow after
        this, and there's no point in further optimizations.

        Applying softmax would convert to a probability, so it's not included in the network. Instead, the softmax
        activation is applied separately, and only when needing to convert into into probabilities for action selection.

        Args:
            state (Numpy array): The state matrix, which I think is...(batch_size, num_states).
        Returns:
            The action-values approximation (Numpy array) of shape (batch_size, num_actions).
        """
        # State shape: (batch_size, state_dims) = (batch_size   , layer_size[0])
        # psi & x shape:(batch_size,num_hidden) = (batch_size   , layer_size[1])
        # W0 shape: (state_dims , num_hidden)   = (layer_size[0], layer_size[1])
        # b0 shape: (1          , num_hidden)   = (1            , layer_size[1])
        # W1 shape: (num_hidden , num_actions)  = (layer_size[1], layer_size[2])
        # b1 shape: (1          , num_actions)  = (1            , layer_size[2])
        # q_vals shape:(batch_size, num_actions)= (batch_size   , layer_size[2])
        W0, b0 = self._weights[0]['W'], self._weights[0]['b']  # weights for layer 0, shape:
        psi = np.dot(state, W0) + b0  # Linear vector result of layer 0 (not activated yet)
        x = np.maximum(psi, 0)  # activation of layer 0, shape()

        W1, b1 = self._weights[1]['W'], self._weights[1]['b']  # weights of layer 1
        q_vals = np.dot(x, W1) + b1  # linear combination of layer 1
        return q_vals

    def check_config_match(self, approximator):
        """
        Check if key parameters are consistent, such that a loaded dataset is compatible. Basically the network shape:
        L1, L2, .., Ln
        :param approximator: The approximator whose config is being compared.
        :return:
        """
        try:
            structure_match = np.all(self.layer_sizes == approximator.layer_sizes)
            assert(approximator._weights) # Just checking that it exists
        except Exception as err:
            print("Failed config check. Structure of current approximator is inconsistent with structure of loading"
                  "approximator.")
            return False
        return structure_match

    def load(self, load_network):
        """
        Load the parameters that are needed for agent memory. Yes, we are violating privacy here, and I really should
        make public facing read-only properties. Maybe later, but it's low priority.
        :param load_network: Network being loaded.
        :return: None
        """
        try:
            self._weights = load_network._weights
        except Exception as err:
            raise("Error: Attempted to load incompatible configuration.", err)
