import numpy as np

def argmax(values):
    """
    Takes in a list of values and returns the index of the item with the highest value.
    Unlike np.argmax, this function breaks ties randomly, encouraging exploration.
    However, it is 3-4x slower than np.argmax.

    returns: int - the index of the highest value in q_values
    """

    values = np.array(values)
    max_val = values.max()
    indices = []
    for i, value in enumerate(values):  # not sure why we don't just enumerate, but ok...
        if value == max_val:
            indices.append(i)
    return np.random.choice(indices)


def one_hot(state, num_states):
    """
    Given num_state and a state, return the one-hot encoding of the state
    """
    # Create the one-hot encoding of state
    # one_hot_vector is a numpy array of shape (1, num_states)

    one_hot_vector = np.zeros(num_states)
    one_hot_vector[int(state)] = 1

    return one_hot_vector


if __name__ == "__main__":
    np.random.seed(0)
    assert np.all(argmax([1,2,5,3,5,4,5,]) == 2)
    assert np.all(argmax([1,2,5,3,5,4,5,]) == 4)

    print("RLHelpers tests pass.")


    import time
    tic = time.time()
    for i in range(10000):
        np.argmax([1,2,5,3,5,4,5,])
    toc = time.time()
    print("numpy time: ", toc-tic)

    tic = time.time()
    for i in range(10000):
        argmax([1,2,5,3,5,4,5,])
    toc = time.time()
    print("custom time: ", toc-tic)


    # Test one_hot
    encoding = one_hot(3, 10)
    expected = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    print(encoding)
    print(expected)
    assert np.all(encoding == np.array(expected))