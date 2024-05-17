import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import unittest

from neurostatx.statistics.utils import KNNimputation, apply_various_models
from neurostatx.utils.factor import efa, cfa
from neurostatx.utils.preprocessing import compute_pca


class TestImputation(unittest.TestCase):

    def test_impute_nans(self):
        # Define a reference dataset.
        ref_df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0, 5.0],
                               'B': [5.0, 4.0, 3.0, 2.0, 1.0],
                               'C': [1.0, 2.0, 3.0, 4.0, 5.0]})

        # Define a dataset to impute.
        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 4.0, 5.0],
                           'B': [5.0, 4.0, 3.0, 2.0, 1.0],
                           'C': [1.0, 2.0, 3.0, 4.0, 5.0]})

        # Impute the dataset.
        out = KNNimputation(ref_df, df, n_neighbors=1)

        # Compare the output with the expected output.
        pd.testing.assert_frame_equal(out, ref_df)

    def test_apply_efa_model(self):
        # Fetch the iris dataset.
        np.random.seed(123)
        df = pd.DataFrame(np.random.rand(100, 10))
        method = "minres"
        nfactors = 2
        rotation = 'varimax'

        # Call function
        efa_mod, ev, v, scores, loadings, communalities = \
            efa(df, method, rotation, nfactors)

        # Apply model.
        transform = apply_various_models(df, efa_mod)

        # Perform assertions
        np.testing.assert_array_equal(scores, transform)

    def test_apply_cfa_model(self):
        # Define test data
        df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
        df.columns = ['sepal_length', 'sepal_width', 'petal_length',
                      'petal_width']
        model = ""
        model += "efa1 =~ sepal_length + petal_length\n"
        model += "efa2 =~ sepal_width + petal_length"

        # Call function
        cfa_mod, scores, stats = cfa(
            df, model)

        # Apply model.
        transform = apply_various_models(df, cfa_mod)

        # Perform assertions
        pd.testing.assert_frame_equal(scores, transform)

    def test_apply_sklearn_model(self):
        # Define test data
        X = np.random.rand(100, 5)
        n_components = 3

        # Call the function with sample data
        X_transformed, model, exp_var, components, p_value, kmo_model = \
            compute_pca(X, n_components)

        # Apply model.
        transform = apply_various_models(X, model)

        # Perform assertions
        np.testing.assert_array_equal(X_transformed, transform)


if __name__ == '__main__':
    unittest.main()
