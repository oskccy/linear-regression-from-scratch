import pandas as pd
import numpy as np

house_dataset = pd.read_csv("housing.csv")

area = house_dataset["area"]
price = house_dataset["price"]

x = np.array(area).reshape(-1, 1)
y = np.array(price).reshape(-1, 1)

house_dataset = [x, y]
