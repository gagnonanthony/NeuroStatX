import os
import tempfile
import unittest
import warnings

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from neurostatx.clustering.fuzzy import (
    FuzzyCMeans,
    fuzzyCmeans,
    search_fuzzy_cmeans,
)
from neurostatx.clustering.viz import plot_fuzzy_cmeans_solutions


class TestFuzzyCMeans(unittest.TestCase):

    def setUp(self):
        self.centroids = np.array([[1, 1], [3, 3], [5, 5]])
        np.random.seed(0)
        self.X = np.concatenate([
            self.centroids[i] + 0.2 * np.random.randn(200, 2)
            for i in range(len(self.centroids))
        ])

    def test_fit_recovers_centroids(self):
        fcm = FuzzyCMeans(n_clusters=3, random_state=0)
        fcm.fit(self.X)

        recovered = np.sort(fcm.cluster_centers_, axis=0)
        for i in range(len(self.centroids)):
            self.assertTrue(np.allclose(
                recovered[i], self.centroids[i], atol=0.1
            ))

        self.assertEqual(fcm.n_features_in_, 2)
        self.assertEqual(fcm.labels_.shape, (self.X.shape[0],))
        self.assertEqual(fcm.u_.shape, (3, self.X.shape[0]))
        check_is_fitted(fcm, "cluster_centers_")

    def test_predict_matches_labels(self):
        fcm = FuzzyCMeans(n_clusters=3, random_state=0)
        fcm.fit(self.X)
        np.testing.assert_array_equal(fcm.predict(self.X), fcm.labels_)

    def test_predict_proba_rows_sum_to_one(self):
        fcm = FuzzyCMeans(n_clusters=3, random_state=0)
        fcm.fit(self.X)
        membership = fcm.predict_proba(self.X)
        self.assertEqual(membership.shape, (self.X.shape[0], 3))
        np.testing.assert_allclose(membership.sum(axis=1), 1.0, rtol=1e-5)

    def test_predict_unfitted_raises(self):
        fcm = FuzzyCMeans(n_clusters=3)
        with self.assertRaises(NotFittedError):
            fcm.predict(self.X)
        with self.assertRaises(NotFittedError):
            fcm.predict_proba(self.X)

    def test_fit_predict(self):
        fcm = FuzzyCMeans(n_clusters=3, random_state=0)
        labels = fcm.fit_predict(self.X)
        np.testing.assert_array_equal(labels, fcm.labels_)

    def test_get_params_set_params(self):
        fcm = FuzzyCMeans(n_clusters=3, m=2.0)
        params = fcm.get_params()
        self.assertEqual(params["n_clusters"], 3)
        self.assertEqual(params["m"], 2.0)
        fcm.set_params(n_clusters=4, m=1.5)
        self.assertEqual(fcm.n_clusters, 4)
        self.assertEqual(fcm.m, 1.5)

    def test_predict_wrong_n_features_raises(self):
        fcm = FuzzyCMeans(n_clusters=3, random_state=0)
        fcm.fit(self.X)
        with self.assertRaises(ValueError):
            fcm.predict(np.ones((10, 5)))


class TestSearchFuzzyCmeans(unittest.TestCase):

    def setUp(self):
        self.centroids = np.array([[1, 1], [3, 3], [5, 5]])
        np.random.seed(0)
        self.X = np.concatenate([
            self.centroids[i] + 0.2 * np.random.randn(200, 2)
            for i in range(len(self.centroids))
        ])

    def test_search_returns_one_solution_per_k(self):
        cntr, u, wss, fpc, ss, chi, dbi, gap, sk = search_fuzzy_cmeans(
            self.X,
            min_clusters=2,
            max_clusters=3,
            n_jobs=1,
            compute_gap=False,
            output=None,
            random_state=0,
        )
        n_solutions = 3 - 2 + 1
        self.assertEqual(len(cntr), n_solutions)
        self.assertEqual(len(u), n_solutions)
        self.assertEqual(len(wss), n_solutions)
        self.assertEqual(len(fpc), n_solutions)

        recovered = np.sort(cntr[1], axis=0)
        for i in range(len(self.centroids)):
            self.assertTrue(np.allclose(
                recovered[i], self.centroids[i], atol=0.1
            ))

    def test_search_invalid_k_range_raises(self):
        with self.assertRaises(ValueError):
            search_fuzzy_cmeans(self.X, min_clusters=5, max_clusters=2)

    def test_deprecated_fuzzyCmeans_wrapper(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cntr, _, _, _, _, _, _, _, _ = fuzzyCmeans(
                self.X, max_cluster=3, output=None
            )
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught)
        )
        self.assertEqual(len(cntr), 2)


class TestPlotFuzzyCmeansSolutions(unittest.TestCase):

    def test_writes_figure(self):
        import matplotlib
        matplotlib.use("Agg", force=True)

        rng = np.random.RandomState(0)
        x = rng.randn(20)
        y = rng.randn(20)
        labels = np.zeros(20, dtype=int)
        labels[10:] = 1
        cntr = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = (
            2, 10, 0.8, cntr, None, None, None, None, None, None, None,
            x, y, labels,
        )
        with tempfile.TemporaryDirectory() as tmp:
            plot_fuzzy_cmeans_solutions(
                [result],
                output=tmp,
                min_clusters=2,
                max_clusters=2,
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(tmp, "viz_multiple_cluster_nb.png")
                )
            )


if __name__ == "__main__":
    unittest.main()
