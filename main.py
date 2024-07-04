import numpy as np
import matplotlib.pyplot as plt

from linear_creator import linear_data
from data_loader import house_dataset
from minmax_normalizer import minmax_normalize, minmax_denormalize

xpoints_u = np.array(house_dataset[0])
ypoints_u = np.array(house_dataset[1])

xpoints, ypoints = minmax_normalize(xpoints_u, ypoints_u)

m = 0
b = 0
alpha_rate = 0.01
epochs = 10000

for e in range(epochs):
    m_gradient = 0
    b_gradient = 0
    n = len(xpoints)
    print(f"EPOCHS TRAINED: {e}")
    for i in range(n):
        x = xpoints[i]
        y = ypoints[i]
        guess_val = (m * x) + b
        error = y - guess_val
        # partial derivative in terms of m
        m_gradient += -2 * x * error
        # partial derivative in terms of b
        b_gradient += -2 * error

    m = m - (m_gradient / n) * alpha_rate
    b = b - (b_gradient / n) * alpha_rate

m, b = minmax_denormalize(xpoints_u, ypoints_u, m, b)

print(f"PREDICTED SLOPE: {m}")
print(f"PREDICTED Y-INTERCEPT: {b}")
linear_data_array = linear_data(m, b, house_dataset)

plt.xlabel("House Area (sqft)")
plt.ylabel("House Price ($US)")
plt.plot(house_dataset[0], house_dataset[1], 'o')
plt.plot(linear_data_array[0], linear_data_array[1])
plt.show()
