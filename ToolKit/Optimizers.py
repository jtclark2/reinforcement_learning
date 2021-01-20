import numpy as np

class Adam:
    def __init__(self):
        pass

    def optimizer_init(self, optimizer_info):
        """
        Set parameters needed to setup the Adam algorithm.
        """
        self.num_inputs = optimizer_info.get("num_inputs")
        self.num_hidden_layer = optimizer_info.get("num_hidden_layer")
        self.num_hidden_units = optimizer_info.get("num_hidden_units")

        # Specify Adam algorithm's hyper parameters
        self.step_size = optimizer_info.get("step_size")
        self.beta_m = optimizer_info.get("beta_m")
        self.beta_v = optimizer_info.get("beta_v")
        self.epsilon = optimizer_info.get("epsilon")

        self.layer_size = np.array([self.num_inputs, self.num_hidden_units, 1])

        # Initialize Adam algorithm's m and v
        self.m = [dict() for i in range(self.num_hidden_layer + 1)]
        self.v = [dict() for i in range(self.num_hidden_layer + 1)]

        for i in range(self.num_hidden_layer + 1):
            # Initialize self.m[i]["W"], self.m[i]["b"], self.v[i]["W"], self.v[i]["b"] to zero
            self.m[i]["W"] = np.zeros((self.layer_size[i], self.layer_size[i + 1]))
            self.m[i]["b"] = np.zeros((1, self.layer_size[i + 1]))
            self.v[i]["W"] = np.zeros((self.layer_size[i], self.layer_size[i + 1]))
            self.v[i]["b"] = np.zeros((1, self.layer_size[i + 1]))

        # Initialize beta_m_product and beta_v_product to be later used for computing m_hat and v_hat
        self.beta_m_product = self.beta_m
        self.beta_v_product = self.beta_v

    def update_weights(self, weights, g):
        """
        Given weights and update g, return updated weights
        """

        for i in range(len(weights)):
            for param in weights[i].keys():
                ### update self.m and self.v
                self.m[i][param] = self.beta_m * self.m[i][param] + (1 - self.beta_m) * g[i][param]
                self.v[i][param] = self.beta_v * self.v[i][param] + (1 - self.beta_v) * (g[i][param] * g[i][param])

                ### compute m_hat and v_hat
                m_hat = self.m[i][param] / (1 - self.beta_m_product)
                v_hat = self.v[i][param] / (1 - self.beta_v_product)

                ### update weights
                weights[i][param] += self.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)

        ### update self.beta_m_product and self.beta_v_product
        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v

        return weights

class SGD:
    def __init__(self):
        pass

    def optimizer_init(self, optimizer_info):
        """
        Set parameters needed to setup the stochastic gradient descent method.
        Assume optimizer_info dict contains: {step_size: float}
        """
        self.step_size = optimizer_info.get("step_size")

    def update_weights(self, weights, g):
        """
        Given weights and update g, return updated weights
        """
        for i in range(len(weights)):
            for param in weights[i].keys():
                weights[i][param] = weights[i][param] + self.step_size * g[i][param]

        return weights
