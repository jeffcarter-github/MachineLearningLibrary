import numpy as np
from scipy import stats


class KMeans(object):
    def __init__(self, n_clusters, init='kmeans++', max_iter=300,
                 random_state=None):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state

        self.n_samples = None
        self.n_features = None

        self.cluster_centers = None
        self.inertia = None

        if self.n_clusters < 2:
            raise ValueError()
        if self.init not in ['forgy', 'random', 'kmeans++']:
            raise ValueError()
        if self.max_iter <= 0:
            raise ValueError()
        if self.random_state:
            if not isinstance(self.random_state, int):
                raise ValueError()
            else:
                np.random.seed(self.random_state)

    def _check_data(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError()
        else:
            self.n_samples, self.n_features = X.shape
            if self.n_clusters > self.n_samples:
                raise ValueError()

    def _init_clusters(self, X):
        if self.init == 'forgy':
            idx = np.random.randint(0, self.n_samples, self.n_clusters)
            self.cluster_centers = X[idx, :]
        elif self.init == 'random':
            assignments =\
                np.argmin(np.random.rand(self.n_clusters, self.n_samples), axis=0)
            self.cluster_centers =\
                np.array([np.average(X[assignments == i], axis=0)
                          for i in range(self.n_clusters)])
        elif self.init == 'kmeans++':
            c_k = np.random.choice(range(X.shape[0]), 1)
            self.cluster_centers = [X[c_k]]
            for i in range(1, self.n_clusters):
                x_k = range(self.n_samples)
                p_k = np.square(np.min([self._calculate_distances(X, center) for center in self.cluster_centers], axis=0))
                p_k = p_k / np.sum(p_k)
                c_k = stats.rv_discrete(values=(x_k, p_k)).rvs(size=1)
                self.cluster_centers.append(X[c_k])
            self.cluster_centers = np.array(self.cluster_centers)

    def _assign_data(self, X):
        distances = []
        for cluster_center in self.cluster_centers:
            distances.append(self._calculate_distances(X, cluster_center))
        distances = np.array(distances)
        return np.argmin(distances, axis=0)

    def _calculate_distances(self, X, point):
        return np.sqrt(np.sum(np.square(point - X[:, ]), axis=1))

    def _calculate_inertia(self, X):
        labels = self._assign_data((X))
        points = self.cluster_centers[labels]
        self.inertia = np.sum(self._calculate_distances(X, self.cluster_centers[labels]))

    def _calculate_centers(self, X, labels):
        centers = []
        for cluster in range(self.n_clusters):
            center = np.average(X[labels == cluster], axis=0)
            if np.sum(np.isfinite(center)==False) != 0:
                center = self.cluster_centers[cluster]
            centers.append(center)
        self.cluster_centers = np.array(centers)

    def fit(self, X):
        self._check_data(X)
        self._init_clusters(X)
        for i in range(self.max_iter):
            idx = self._assign_data(X)
            self._calculate_centers(X, idx)
        self._calculate_inertia(X)

