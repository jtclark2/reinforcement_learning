import numpy as np

# Initial conditions
class Averager:
    def __init__(self):
        # initialize to 0. Wouldn't make sense to have momentum before we start
        self.param = 0

        # Decay rate (halflife of ~1/(1-b)...eg: b=0.9 --> 1/(1-0.9)=1/.1=10 steps to decay. 1 would never decay
        self.b = 0.9
        self.b_pow = 1

    def reset(self):
        self.param = 0

    def update_param(self, param, remove_bias=True):
        self.param = self.b * self.param + (
                    1 - self.b) * param  # not sure why param coefficient=(1-b)...that's usually ~1 anyways
        # Which is the same as the incremental update equation (prove it with some simple algebra):
        #             = (self.b-1+1)*self.param + (1-self.b)*param
        #             = (self.b-1)*self.param + (1-self.b)*param +self.param
        #             = -(1-self.b)*self.param + (1-self.b)*param +self.param
        #             = (1-self.b)*(-self.param) + (1-self.b)*param +self.param
        #             = (1-self.b)*(param-self.param) + self.param
        #             = self.param + (1 - self.b) * (param - self.param)
        #               where 1-self.b is defined as alpha in the standard incremental update
        if remove_bias:
            self.b_pow *= self.b
            output = self.param / (1 - self.b_pow)
        else:
            output = self.param
        return output

    def average_entire_list(self, vector, full_range=None):
        if full_range is None:
            full_range = 1 / (1 - self.b)
            half_range = int(full_range // 2)  # [-1/2 range, +1/2 range]
        average = [np.average(vector[max(i - half_range, 0):min(i + half_range, len(vector))]) for i, _ in
                   enumerate(vector)]
        return average

    def kalman_filter(self):
        raise NotImplementedError

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import random
    opt = Averager()
    index = [i for i in range(100)]
    actual = [random.random() * 10]
    for i in range(99):
        new_value = (random.random() - 0.5)
        actual.append(new_value + actual[-1])

    opt.reset()
    average = [opt.update_param(i, False) for i in actual]

    opt.reset()
    average_unbiased = [opt.update_param(i) for i in actual]

    full_list_ave = opt.average_entire_list(actual)

    plt.plot(index, actual, 'k', index, average, 'b', index, average_unbiased, 'g', index, full_list_ave, 'y')
    plt.legend(('actual', 'average', 'unbiased_average'))
    plt.show()