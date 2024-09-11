# Regresion sobre cuatro puntos usando tres nodos, para poder usar sólo dos
# nodos (n - 2) se necesita activar la pendiente negativa en las neuronas ReLU

import matplotlib.pyplot as plt
import numpy as np
import keras

X = np.array([0,0.2,0.4,0.6]).reshape(-1, 1)
y = np.array([1,0,1,0]).reshape(-1, 1)

# Define a minimal model
model = keras.models.Sequential()
model.add(keras.layers.Input((1,)))
# Hidden layers
model.add(keras.layers.Dense(3, activation='relu', name='hidden'))
# Output layer
model.add(keras.layers.Dense(1, activation='linear', name='output'))

kernel = np.array([2,0,1]).reshape((3,1,1)) #(size, input_channels, output_channels)
bias = np.zeros((1,)) # 1 channel

#%%

model.get_layer(index=0).set_weights([np.array([1,1,1]).reshape(-1, 3), np.array([0,-0.2,-0.4])])
model.get_layer(index=1).set_weights([np.array([-5,10,-10]).reshape(-1, 1), np.array([1])])

# model.compile(loss='mean_squared_error',
#               optimizer='adam')
# model.fit(
#     X,
#     y,
#     epochs=100,
# )

plt.plot(X, y)
X_test = np.linspace(0, 0.6, 100)
y_nn = model.predict(X_test.reshape(-1, 1))
plt.plot(X_test, y_nn.reshape(-1), color='red')
plt.show()

