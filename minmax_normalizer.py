import numpy as np

# returns two normalized arrays (range [0, 1])


def minmax_normalize(x, y):
    x_min = np.min(x)
    y_min = np.min(y)
    x_max = np.max(x)
    y_max = np.max(y)
    # vectorized operation avoids iteration
    return (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)


def minmax_denormalize(x, y, m, b):
    x_min = np.min(x)
    y_min = np.min(y)
    x_max = np.max(x)
    y_max = np.max(y)
    # applies reverse factors to output
    m = m * (y_max - y_min) / (x_max - x_min)
    b = b * (y_max - y_min) + y_min - m * x_min

    return m, b
