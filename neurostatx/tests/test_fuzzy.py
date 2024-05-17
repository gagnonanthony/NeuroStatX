import unittest
import numpy as np
from neurostatx.clustering.fuzzy import fuzzyCmeans


class TestFuzzyCmeans(unittest.TestCase):

    def setUp(self):
        # Define three centroids
        self.centroids = np.array([[1, 1], [3, 3], [5, 5]])

        # Generate data around the centroids
        np.random.seed(0)
        self.X = np.concatenate([
            self.centroids[i] + 0.2 * np.random.randn(200, 2)
            for i in range(len(self.centroids))
        ])

    def test_fuzzyCmeans(self):
        # Test the function with generated data
        cntr, _, _, _, _, _, _, _, _ = fuzzyCmeans(self.X, max_cluster=3)

        # Check if the returned centroids are close to the initially chosen
        # centroids
        for i in range(len(self.centroids)):
            self.assertTrue(np.allclose(np.sort(cntr[1], axis=0)[i],
                                        self.centroids[i],
                                        atol=0.1))


if __name__ == '__main__':
    unittest.main()
