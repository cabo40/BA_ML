
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA

iris = load_iris()
X = iris.data
y = iris.target

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

n_components = 4
pca = PCA(n_components=n_components)
pca.fit(X)
X_pca = pca.transform(X)

np.var(X[:,0])
np.var(X[:,1])
np.var(X[:,2])
np.var(X[:,3])

np.var(X_pca[:,0])
np.var(X_pca[:,1])

plt.scatter(X_pca[:,0], X_pca[:,1], c=y)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()


pca.fit_transform(X)


colors = ["navy", "turquoise", "darkorange"]

for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        plt.scatter(
            X_transformed[y == i, 0],
            X_transformed[y == i, 1],
            color=color,
            lw=2,
            label=target_name,
        )

    if "Incremental" in title:
        err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
        plt.title(title + " of iris dataset\nMean absolute unsigned error %.6f" % err)
    else:
        plt.title(title + " of iris dataset")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.axis([-4, 4, -1.5, 1.5])

plt.show()
