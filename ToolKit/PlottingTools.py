"""
This contains a handful of plotting tools, using matplotlib. Nothing revolutionary, just for convenience.
"""
import numpy as np
import matplotlib.pyplot as plt

class PlottingTools:

    @classmethod
    def plot_action_value_2d(cls, value_approximator, resolution=50):
        min1 = value_approximator._state_boundaries[0][0]
        max1 = value_approximator._state_boundaries[0][1]
        step_size_1 = (max1-min1)/resolution
        min2 = value_approximator._state_boundaries[1][0]
        max2 = value_approximator._state_boundaries[1][1]
        step_size_2 = (max2-min2)/resolution

        variable1 = np.zeros((resolution, resolution))
        variable2 = np.zeros((resolution, resolution))
        value = np.zeros((resolution, resolution))

        for index1 in range(resolution):
            var1 = (index1*step_size_1) + min1
            for index2 in range(resolution):
                var2 = (index2*step_size_2) + min2

                variable1[index1, index2] = var1
                variable2[index1, index2] = var2
                value[index1,index2] = np.average(value_approximator.get_values(np.array([var1, var2])))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(variable1, variable2, value)
        plt.show()

    @classmethod
    def plot_smooth(cls, results, smoothing = 100):
        smoothing = min(len(results)//2, smoothing)
        running_avg = [np.average(results[x:x+smoothing]) for x, _ in enumerate(results[:-smoothing])]
        x_axis = np.array([x for x, _ in enumerate(results)])
        plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(running_avg)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward per Episode")
        plt.yscale("linear")
        plt.show()

    @classmethod
    def multiline_plot(cls, x, y1, y2):
        plt.plot(x, y1, 'r', x, y2, 'b')
        plt.show()