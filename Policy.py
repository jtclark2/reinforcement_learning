import random

class Discrete_Explicit_Policy:
    def __init__(self):
        """
        Policy for discrete action space.
        """
        pass

    def policy(self, state):
        """
        Calculates the policy.

        Should poicy be passed in instead? The format will vary for dicrete/continuous action spaces, but I
        want flexibility to switch between types of exploration: softmax, epsilon-greedy/epsilon-soft,

        :param state: The state for which the policy should be calculated.
        :return: policy, an np.array of action probabilities. The indices correspond to the respective actions,
            such that policy[action] = chance_of_occurance, and policy[:].sum() = 1
        """
        raise NotImplementedError()

    def select_action(self, state):
        """
        Select an action for the given state, based on the current policy.

        This solves the _policy() format question. Don't extend this method to use expectation/max for Sarsa/Q-learning.
        Those will use policy directly or create their own private methods.

        :return: action (selected randomly) for the input state and current policy.
        """
        action = random.choice(self._policy(state))
        return action

    def select_greedy_action(self):
        """
        Select the greedy action, the best action (according to current policy).
        :return:
        """
        action = random.choice(self._policy(state))


class Implicit_Policy:
    def policy(self, values):

class Continuous_Policy:
    """
    A continuous function approximation. This is distinguished from discrete because the data structures change from
    simple vectors being passed around, to custom functions.
    """