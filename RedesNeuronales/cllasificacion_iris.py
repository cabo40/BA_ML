from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Convert labels to categorical format
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 categorias

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train,
          y_train_cat,
          epochs=100,
          validation_data=(X_test, y_test_cat))

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred) # Clases calculadas
plt.show()

plt.scatter(X_test[:,0], X_test[:,1], c=y_test) # Clases reales
plt.show()
