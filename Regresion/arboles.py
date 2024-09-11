import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor

def f(x):
    return np.sin(x)

np.random.seed(0)
x = np.linspace(0, 4*np.pi, 50)
y = f(x)
plt.scatter(x, y)
plt.show()

tree = DecisionTreeRegressor(max_depth=5)
tree.fit(x.reshape(-1,1), y)
y_pred = tree.predict(x.reshape(-1, 1))

plt.scatter(x, y_pred)
plt.show()