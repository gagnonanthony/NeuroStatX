import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from neurostatx.clustering.metrics import (compute_evaluation_metrics,
                                           compute_knee_location,
                                           compute_sse,
                                           compute_gap_stats,
                                           find_optimal_gap,
                                           compute_rand_index)


class TestComputeEvaluationMetrics(unittest.TestCase):

    def setUp(self):
        # Generate synthetic data with known characteristics
        self.X, true_labels = make_blobs(n_samples=300, centers=3,
                                         cluster_std=0.5,
                                         random_state=42)

        # Assume known silhouette score, Calinski-Harabasz index, and
        # Davies-Bouldin index values
        self.expected_ss = 0.92
        self.expected_chi = 20500.0
        self.expected_dbi = 0.10

        # Assuming labels are known in this synthetic data
        self.labels = true_labels

    def test_compute_evaluation_metrics(self):
        # Test the function with synthetic data and known characteristics
        ss, chi, dbi = compute_evaluation_metrics(self.X, self.labels)

        # Check if the returned values are close to the expected values
        self.assertAlmostEqual(ss, self.expected_ss, places=2)
        self.assertAlmostEqual(chi, self.expected_chi, places=-3)
        self.assertAlmostEqual(dbi, self.expected_dbi, places=1)


class TestComputeKneeLocation(unittest.TestCase):

    def setUp(self):
        # Elbow data.
        elbow_point = 5
        self.values = np.zeros(10)

        # Generate data before the elbow point with a linear relationship
        self.values[:elbow_point] = -2 * np.arange(1, elbow_point + 1)

        # Generate data after the elbow point with a different slope
        self.values[elbow_point:] = -0.5 * np.arange(1, 10 - elbow_point + 1)

        # Add random noise to the data
        self.values += np.random.randn(10) * 0.1

    def test_compute_knee_location(self):
        # Check if the returned elbow location is close to the expected value
        elbow = compute_knee_location(self.values)
        self.assertEqual(elbow - 1, 5)


class TestComputeSSE(unittest.TestCase):

    def setUp(self):
        # Generate random centroids
        self.centroids = np.random.rand(3, 2)

        # Initialize variables
        iteration = 0
        scale_factor = 1.0
        self.target_wss = 100.0

        self.data = np.zeros((1000, 2))
        self.cluster_assignments = np.random.randint(0, 3, size=1000)
        # Repeat until the difference between actual and target WSS falls
        # within the tolerance or max iterations are reached
        while True:
            # Generate data points around centroids
            for i in range(1000):
                centroid_index = self.cluster_assignments[i]
                centroid = self.centroids[centroid_index]
                # Adjust covariance matrix for tighter clusters
                covariance_matrix = np.eye(2) * 0.1
                self.data[i] = np.random.multivariate_normal(centroid,
                                                             covariance_matrix)

            # Compute actual WSS
            self.actual_wss = np.sum(
                np.linalg.norm(
                    self.data - self.centroids[self.cluster_assignments],
                    axis=1) ** 2)

            # Check if the actual WSS falls within the desired range
            if abs(self.actual_wss - self.target_wss) \
                    <= 10.0 or iteration >= 500:
                break

            # Adjust the scale factor based on the difference between actual
            # and target WSS
            scale_factor *= np.sqrt(self.target_wss / self.actual_wss)
            self.data *= scale_factor

            # Increment iteration counter
            iteration += 1

    def test_compute_see(self):
        # Check if the returned WSS is close to the expected value.
        wss = compute_sse(self.data, self.centroids, self.cluster_assignments)
        self.assertAlmostEqual(wss, self.actual_wss, places=2)


class TestComputeGapStats(unittest.TestCase):

    def setUp(self):
        # Simulate gap values.
        self.gap_values = [0.1, 0.2, 0.3, 0.2, 0.1]
        self.gap_std_devs = [0.01, 0.02, 0.03, 0.02, 0.01]

        # Simulate data for gap statistic.
        np.random.seed(0)
        self.X = np.random.rand(1000, 2)
        self.wss = 100.0

    def test_find_optimal_gap(self):
        # Test the function with simulated data
        optimal_k = find_optimal_gap(self.gap_values, self.gap_std_devs)
        self.assertEqual(optimal_k, 2)

    def test_compute_gap_stats(self):
        # Test the function with simulated data
        np.random.seed(0)
        gap, _ = compute_gap_stats(self.X, self.wss, 1000, 3)

        # Assert that the computed gap statistic is equal or almost equal to
        # the expected value
        self.assertGreaterEqual(gap, -0.45)
        self.assertLessEqual(gap, -0.40)


class TestComputeRandIndex(unittest.TestCase):

    def setUp(self):
        # Generate synthetic data with known clustering assignments
        num_samples = 100
        num_clusters = 3
        self.clustering_dict = {}
        for i in range(num_clusters):
            self.clustering_dict[f'Cluster_{i}'] = \
                pd.DataFrame(np.random.rand(num_samples, num_clusters))

        # Calculate the expected Rand Index
        self.expected_ari = np.zeros((num_clusters, num_clusters))
        for i, (_, arr1) in enumerate(self.clustering_dict.items()):
            for j, (_, arr2) in enumerate(self.clustering_dict.items()):
                labels1 = np.argmax(arr1, axis=1)
                labels2 = np.argmax(arr2, axis=1)
                self.expected_ari[i, j] = adjusted_rand_score(labels1, labels2)

    def test_compute_rand_index(self):
        # Test the function with synthetic data
        ari = compute_rand_index(self.clustering_dict)

        # Compare the computed Rand Index with the expected value
        np.testing.assert_array_almost_equal(ari, self.expected_ari)


if __name__ == '__main__':
    unittest.main()
