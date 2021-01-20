import numpy as np
import RLHelpers
import pickle
import os

class SimpleNNAgent:
    """
    This class uses a simple neural network (1 hidden layer). It is intentionally simplified, but as a result, lacks
     the flexibility to define/modify the network.

    Most of the methods were copied from my HW assignment #2 in an RL series in coursera,
    "Prediction and Control Function Approximtation". I say this both to give credit to the professors, who wrote a
    template for the code, and also to explain some of the differences in style and naming convention.
    In writing the agent in that course, I felt the training wheels were a bit too restrictive, so I'm trying to
    adapt that to new environments, and restructuring a bit. Ultimately, this is a stepping stone towards implementing
    a more elaborate agent that will use tensorflow for more extensible network design.
    """

    #################### Initialization ####################
    def __init__(self, agent_info={}):
        self.name = "SimpleNNAgent"

        # Set attributes according to agent_info
        self.num_observations = agent_info.get("num_observations") # eg: 2 for the mountain-car (pos, vel), 4 for CartPole-v1, etc.
        self.num_actions = agent_info.get("num_actions") # eg: 2 for the mountain-car (pos, vel), 4 for CartPole-v1, etc.
        self.num_inputs = self.num_observations*self.num_actions
        # just 1 for now, but a bit of flexibility is built in
        # layer size is initialized a few lines down and currently expects this to be =1
        self.num_hidden_layer = agent_info["num_hidden_layer"] # 1
        self.num_hidden_units = agent_info["num_hidden_units"] # 100  # how many nodes in the hidden layer
        self.discount_factor = agent_info["discount_factor"] # 1
        self.epsilon = agent_info["epsilon"] # 0.01

        ### Define the neural network's structure
        # Note this is the number of nodes of each layer as an array (input size, hidden size, output size)
        # Inputs (state), hidden, outputs (action-value)
        self.layer_size = np.array([self.num_inputs, self.num_hidden_units, 1])


        # Initialize the neural network's parameters
        # [layer_number]['W' for weights np.array(m,n) or 'b' for weight of constant term]
        self.weights = [dict() for i in range(self.num_hidden_layer + 1)]
        for i in range(self.num_hidden_layer + 1):
            mean = 0
            std_dev = np.sqrt(2 / self.layer_size[i])
            self.weights[i]['W'] = np.random.normal(mean, std_dev, (self.layer_size[i], self.layer_size[i + 1]))
            self.weights[i]['b'] = np.random.normal(mean, std_dev, (1, self.layer_size[i + 1]))

        # Specify the optimizer
        self.optimizer = agent_info["optimizer"]

        self.previous_state = None
        self.previous_action = None

    #################### State Function Approximation ####################
    def get_value(self, state, action):
        """
        Compute value of input s given the weights of a neural network
        """
        state_action_vector = self.get_inputs(state, action) # stack actions [act1_obs1, act1_obs2, act2_obs1, ... ]

        W0 = self.weights[0]["W"]
        b0 = self.weights[0]["b"]
        W1 = self.weights[1]["W"]
        b1 = self.weights[1]["b"]
        psi = np.matmul(state_action_vector, W0) + b0 # input --> layer 1
        x = np.maximum(0, psi) # relu activation
        v = np.matmul(x, W1) + b1 #layer ` activation --> layer 2
        # ----------------
        return v

    def get_gradient(self, state, action):
        """
        Find the gradient of this neural network (fully connected, 1 hidden layer)
        Return the gradient of v with respect to the weights where v is the final output (v=x*W1+b)
        """
        grads = [dict() for i in range(len(self.weights))]

        # Extract weights
        W0 = self.weights[0]["W"]
        b0 = self.weights[0]["b"]
        W1 = self.weights[1]["W"]
        # b1 = self.weights[1]["b"] # don't need the last constant - it's grad = 0
        state_action_vector = self.get_inputs(state,action)

        # Forward propagation
        psi = np.matmul(state_action_vector, W0) + b0
        # print("muliplicative term:" , np.matmul(state_action_vector, W0))
        # print("+b:" , b0)
        x = np.maximum(0, psi)

        # Back prop
        grads[1]["W"] = x.transpose()
        grads[1]["b"] = np.array([[1]])

        # WARNING : SEEMS LIKE IT MIGHT NOT GENERALIZE TO NEW APPLICATION
        # Does this work without 1-hot encoding on input???
        print("X:", x)
        x_rectified = x > 0 # relu
        I = np.identity(x_rectified.shape[1]) # This only works for this specific gradient - will not generalize well
        x_diag = x_rectified * I # identity shape with x_rectified on th diagonal
        grads[0]["W"] = np.matmul(state_action_vector.transpose(), np.dot(W1.transpose(), x_diag))
        grads[0]["b"] = np.dot(W1.transpose(), x_diag)

        for layer in range(2):
            for param in ['W', 'b']:
                # print("grads[%d][%s]: " % (layer, param), grads[layer][param][0:4].transpose())
                pass

        # ----------------

        return grads

    def get_inputs(self, state, action):
        input_encoding = np.zeros((1,self.num_actions*self.num_observations)) # inputs for a given state_action pair
        input_encoding[0,action*self.num_observations:(action+1)*self.num_observations] = state
        return input_encoding

    def get_all_action_values(self, state):
        action_values = [self.get_value(state, action) for action in range(self.num_actions)] # forward propagate
        return np.array(action_values)

    #################### Policy and Action Selection ####################
    def select_action(self, state):
        action_values = self.get_all_action_values(state)

        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(self.num_actions)  # randomly explore
        else:
            chosen_action = RLHelpers.argmax(action_values)  # act greedy
        # ----------------

        return chosen_action, action_values[chosen_action]

    #################### Update Steps (start, step, end) ####################

    def start(self, state):
        """
        The first method called when the experiment starts, called after the environment starts.
        Args:
            state (Numpy array): the state from the environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.previous_state = state
        self.previous_action, _ = self.select_action(state)

        return self.previous_action

    def step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

        ### Compute TD error
        previous_value = self.get_value(self.previous_state, self.previous_action)
        if state is None: # end step (when env says we're done)
            delta = reward - previous_value
        else: # normal step case
            current_action, next_value = self.select_action(state)
            v_optimal = self.get_all_action_values(state).max()
            delta = reward + self.discount_factor * next_value - previous_value # SARSA update
            # delta = reward + self.discount_factor * v_optimal - previous_value # Q AgentMemory Update

        ### Retrieve gradients and compute gradient descent step size
        grads = self.get_gradient(self.previous_state, self.previous_action)

        ### Compute g (1 line)
        g = [dict() for i in range(self.num_hidden_layer + 1)]
        for i in range(self.num_hidden_layer + 1):
            for param in self.weights[i].keys():
                g[i][param] = delta * grads[i][param]

        ### update the weights using self.optimizer
        self.optimizer.update_weights(self.weights, g)

        if state is not None:
            self.previous_state = state
            self.previous_action = current_action

            return self.previous_action

    def end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        return self.step(reward, None)

    #################### Save and Load Data ####################
    def save_learning(self, env_name=""):
        self.save_path = r"./Agents/AgentMemory/%s_%s.p" % (self.name, env_name) # /Agents/AgentMemory"
                         #os.path.dirname(os.path.realpath(__file__))
                         #  os.getcwd()
        print("Saving weights to: ", self.save_path)
        learning = self.weights
        pickle.dump(learning, open( self.save_path, "wb" ) )

    def load_learning(self, env_name=""):
        self.load_path = r"./Agents/AgentMemory/%s_%s.p" % (self.name, env_name)
        if os.path.isfile(self.load_path):
            learning = pickle.load(open(self.load_path, "rb"))
            self.weights = learning
            return True
        else:
            print("Warning: Unable to load file. Program will proceed without loading.")
            # time.sleep(3)
            return False


if __name__ == "__main__":
    from Agents import Optimizers
    import Trainer
    import gym

    ############### Environment Setup (and configuration of agent for env) ###############
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    ############### SimpleNNAgent Setup ###############
    observations = env.reset() # don't really need a reset here, but .n doesn't seem to be defined
    agent_info = { "num_observations": len(observations),
                   "num_actions": env.action_space.n,
                   "num_inputs": len(observations)*env.action_space.n,
                   "num_hidden_layer": 1,
                   "num_hidden_units": 100,
                   "discount_factor": 1, # gamma
                   "epsilon": 0.1, # amount of exploration
                 }

    optimizer_info = {
        "num_inputs": agent_info["num_inputs"],
        "num_hidden_layer": agent_info["num_hidden_layer"],
        "num_hidden_units": agent_info["num_hidden_units"],
        "step_size": 0.1,
        "beta_m": 0.9,
        "beta_v": 0.999,
        "epsilon": 10**-8, # no explode term in ADAM that nobody ever touches (not to be confused with agent epsilon)
    }
    optimizer = Optimizers.Adam()
    optimizer.optimizer_init(optimizer_info)
    agent_info["optimizer"] = optimizer

    agent = SimpleNNAgent.SimpleNNAgent(agent_info)
    load_status = agent.load_agent_memory(env_name)

    ############### Trainer Setup (load run history) ###############
    trainer = Trainer.Trainer(env, agent)
    if(load_status):
        trainer.load_run_history(env_name)

    ############### Define Run inputs and Run ###############
    total_episodes = 200
    max_steps = 1000
    render_interval = 100 # 0 is never
    trainer.train_fixed_steps(total_episodes, max_steps, render_interval) # multiple runs for up to total_steps

    ############### Save to file and plot progress ###############
    agent.save_agent_memory(env_name)
    trainer.save_run_history(env_name)
    Trainer.plot(agent, np.array(trainer.rewards) )
