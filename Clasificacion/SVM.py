from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

y_new =y
y_new[y==2] = 1
plt.scatter(X[:,0], X[:,1], c=y_new)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
model = LinearSVC()
model.fit(X_train, y_train)
y_output = model.predict(X_test)

plt.scatter(X_test[:,0], X_test[:,1], c=y_output) # Clasificación entrenada
plt.show()

plt.scatter(X_test[:,0], X_test[:,1], c=y_test) # Clasificación real
plt.show()
