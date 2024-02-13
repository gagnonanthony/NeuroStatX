import unittest
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
import semopy
from sklearn.datasets import load_iris

from CCPM.utils.factor import (horn_parallel_analysis,
                               apply_efa_only,
                               apply_efa_and_cfa)


class TestFunctions(unittest.TestCase):

    def test_horn_parallel_analysis(self):
        # Not optimal to test this function, but it is a good start.
        # Define test data
        df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
        output_folder = "./"
        method = "minres"
        rotation = None
        nfactors = 4
        niter = 20

        # Call function
        suggfactors, suggcomponents = horn_parallel_analysis(
            df, output_folder, method, rotation, nfactors, niter)

        # Perform assertions
        self.assertEqual(suggfactors, 3)
        self.assertEqual(suggcomponents, 1)

        # Additional assertion: check if the output file exists
        import os
        self.assertTrue(
            os.path.exists(f"{output_folder}/horns_parallel_screeplot.png"))

    def test_apply_efa_only(self):
        # Not optimal to test this function, but it is a good start.
        # Define test data
        np.random.seed(123)
        df = pd.DataFrame(np.random.rand(100, 10))
        method = "minres"
        nfactors = 2
        rotation = 'varimax'

        # Call function
        efa = apply_efa_only(df, method, nfactors, rotation)

        # Perform assertions
        self.assertIsInstance(efa, FactorAnalyzer)

    def test_apply_efa_and_cfa(self):
        # Not optimal to test this function, but it is a good start.
        # Define test data
        df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
        df.columns = ['sepal_length', 'sepal_width', 'petal_length',
                      'petal_width']
        method = "minres"
        nfactors = 2
        rotation = 'varimax'
        train_size = 0.5
        threshold = 0.4
        random_state = 42

        # Call function
        efa, cfa = apply_efa_and_cfa(
            df, method, nfactors, rotation, train_size, threshold,
            random_state)

        # Perform assertions
        self.assertIsInstance(efa, FactorAnalyzer)
        self.assertIsInstance(cfa, semopy.Model)


if __name__ == '__main__':
    unittest.main()
