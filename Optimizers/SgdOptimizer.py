"""
Don't use this...basically never as good as ADAM, and it's the most rudimentary implementation, with no vectorization.
"""
class SGD:
    def __init__(self):
        raise DeprecationWarning("This algorithm is sub-optimal, and there's no reason to use it, except for"
                                 "instructional purposes (ie: look at the concept behind gradient descent.)")
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
