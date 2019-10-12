import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs

def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
            # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels

def cal_variance(x):
    x_bar = np.average(x)
    n = len(x)
    s = np.sum(np.power(x - x_bar, 2)) / (n-1)
    return s

def compute_obj_size(x0, y0, x1, y1):
    w = np.absolute(x1 - x0)
    h = np.absolute(y1 - y0)
    return w, h

# X, y_true = make_blobs(n_samples=300, centers=4,
# cluster_std=0.60, random_state=0)
# centers, labels = find_clusters(X, 4)
# print(labels)