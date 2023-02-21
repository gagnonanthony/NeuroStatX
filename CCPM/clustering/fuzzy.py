import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state


class FuzzyCoClustering:
    def __init__(self, n_clusters=2, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        rng = check_random_state(self.random_state)
        n_samples, n_features = X.shape

        # Initialize cluster membership matrices
        U = rng.rand(self.n_clusters, n_samples)
        V = rng.rand(self.n_clusters, n_features)

        # Normalize cluster membership matrices
        U /= U.sum(axis=0)
        V /= V.sum(axis=0)

        # Compute initial objective function value
        obj = self._compute_objective(X, U, V)

        # Iterate until convergence or max iterations reached
        for i in range(self.max_iter):
            # Update cluster membership matrices
            U, V = self._update_memberships(X, U, V)

            # Normalize cluster membership matrices
            U /= U.sum(axis=0)
            V /= V.sum(axis=0)

            # Compute new objective function value
            new_obj = self._compute_objective(X, U, V)

            # Check for convergence
            if np.abs(obj - new_obj) < 1e-5:
                break

            obj = new_obj

        self.membership_matrix_ = U
        self.cluster_centers_ = V

    def _compute_objective(self, X, U, V):
        # Compute squared Frobenius norm between X and UV'
        return np.sum(U.T.dot(pairwise_distances(X, V, metric='sqeuclidean')).dot(U))

    def _update_memberships(self, X, U, V):
        # Compute numerator and denominator matrices for updating U
        num_U = X.dot(V.T)
        denom_U = U.dot(V).dot(V.T)

        # Apply fuzzy c-means update rule to U
        U_new = U * np.sqrt(num_U / denom_U)

        # Compute numerator and denominator matrices for updating V
        num_V = U.T.dot(X)
        denom_V = U.T.dot(U).dot(V)

        # Apply fuzzy c-means update rule to V
        V_new = V * np.sqrt(num_V / denom_V)

        return U_new, V_new