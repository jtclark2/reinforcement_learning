"""
This agent never will be implemented. This is probably obvious to the more experienced, but to those of you who wanted
to create an agent using policy-iteration/value-iteration, or any variant using dynamic programming with
state-value functions, I'll try to explain.
The short answer is that value-based approaches require a full model of the environment.

The longer answer:
First, let's describe the env function formally. It describes what the result if you start in state S and take action A.
In particular, the new state the agent ends up in, S', and the reward received, R. In a simplified world, the action is
deterministic, so you effectively "know" it. Take a step to the west and go 1 unit to the west. However,
in more realistic problems, this transition has some chance due to real-world variables. So an agent performing the same
action could end up with a distribution of possible outcomes.
(eg: action = move west --> new state = 1+/-.1 units west).

Therefore, you can't say for sure where you'll end up after taking an action. That means that you can't determine the
value of the new state S'. Therefore, you can't determine the values of each action. Therefore, you can't update the
value function at each step according to the Bellman equation. This means the dynamic programming form of value
iteration is impossible unless you know the environment update function, which we generally don't.

I think you could use the Bellman Value Equation (V update) with Monte Carlo method, but it's slow compared to dynamic
solutions, such as TD algorithms...Maybe I'll implement that, but I'm not sure it's worth it.

"""

class AbstractAgent:
    def __init__(self):
        raise NotImplementedError()
