import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

nbrs = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree')

nbrs.fit(X_train, y_train)
y_pred = nbrs.predict(X_test)

plt.scatter(X_test[:,0], X_test[:,1], c=y_pred)
plt.show()
# y_new =y
# y_new[y==2] = 1
# plt.scatter(X[:,0], X[:,1], c=y_new)
# plt.show()


from sklearn.decomposition import PCA, IncrementalPCA

iris = load_iris()
X = iris.data
y = iris.target

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

n_components = 4
pca = PCA(n_components=n_components)
pca.fit(X)
X = pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

nbrs = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree')

nbrs.fit(X_train, y_train)
y_pred = nbrs.predict(X_test)

plt.scatter(X_test[:,0], X_test[:,1], c=y_pred)
plt.show()
