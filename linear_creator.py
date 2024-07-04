# linear prediction creator using slope & intercept
def linear_data(m, b, dataset):
    x_values = [val for val in dataset[0]]
    y_values = [((m * x) + b) for x in x_values]
    return [x_values, y_values]
