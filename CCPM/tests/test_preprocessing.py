import unittest
import pandas as pd
import numpy as np

from CCPM.utils.preprocessing import (merge_dataframes,
                                      compute_pca,
                                      remove_nans,
                                      compute_shapiro_wilk_test,
                                      compute_correlation_coefficient)


class TestFunctions(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        self.df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0],
            'B': [np.nan, 2.0, 3.0, 4.0],
            'C': [1.0, 2.0, 3.0, 4.0]
        })

        self.wilk = pd.DataFrame(np.random.normal(size=3000),
                                 columns=['A'])

    def test_remove_nans(self):
        rows_with_nans, complete_rows = remove_nans(self.df)
        expected_rows_with_nans = pd.DataFrame({
            'A': [1.0, np.nan],
            'B': [np.nan, 3.0],
            'C': [1.0, 3.0]
        })
        expected_rows_complete = pd.DataFrame({
            'A': [2.0, 4.0],
            'B': [2.0, 4.0],
            'C': [2.0, 4.0]
        })

        pd.testing.assert_frame_equal(rows_with_nans.reset_index(drop=True),
                                      expected_rows_with_nans)
        pd.testing.assert_frame_equal(complete_rows.reset_index(drop=True),
                                      expected_rows_complete)

    def test_compute_shapiro_wilk_test(self):
        wilk, pvalues = compute_shapiro_wilk_test(self.wilk)
        self.assertAlmostEqual(wilk[0], 0.999, delta=0.001)

    def test_compute_correlation_coefficient(self):
        corr_mat = np.array([[1.0, 0.8, 0.2],
                             [0.8, 1.0, 0.5],
                             [0.2, 0.5, 1.0]])
        data = pd.DataFrame((
            np.random.multivariate_normal(mean=np.zeros(len(corr_mat)),
                                          cov=corr_mat,
                                          size=1000)),
                            columns=['A', 'B', 'C'])

        # Call the function with sample data
        correlation_matrix = compute_correlation_coefficient(data,
                                                             out_folder='./')

        # Check if the output is a DataFrame
        self.assertIsInstance(correlation_matrix, pd.DataFrame)

        # Check if the dimensions of the correlation matrix are correct
        n_columns = len(self.df.columns)
        self.assertEqual(correlation_matrix.shape, (n_columns, n_columns))

        # Check if the diagonal elements are all 1 (since correlation of a
        # variable with itself is 1)
        self.assertTrue(np.allclose(np.diag(correlation_matrix), 1))

        # Validate if the correlation matrix match the one used to generate the
        # sample data
        np.testing.assert_allclose(correlation_matrix.values,
                                   corr_mat, atol=0.05)

    def test_merge_dataframes(self):
        df1 = pd.DataFrame({'index': [1, 2, 4], 'A': [1, 2, 3],
                            'B': [4, 5, 6]})
        df2 = pd.DataFrame({'index': [1, 2, 4], 'C': [7, 8, 9],
                            'D': [10, 11, 12]})
        dict_df = {'df1': df1, 'df2': df2}
        merged_df = merge_dataframes(dict_df, index='index')
        expected_df = pd.DataFrame({
            'index': [1, 2, 4],
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9],
            'D': [10, 11, 12]
        }).set_index('index')
        pd.testing.assert_frame_equal(merged_df, expected_df)

    def test_compute_pca(self):
        # Hardly any testable assertions can be made for this function, so we
        # will only check if the output variables have the correct data types
        # and dimensions. To be updated in the future.
        X = np.random.rand(100, 5)
        n_components = 3

        # Call the function with sample data
        X_transformed, exp_var, components, p_value, kmo_model = \
            compute_pca(X, n_components)

        # Check if the output variables have the correct data types
        self.assertIsInstance(X_transformed, np.ndarray)
        self.assertIsInstance(exp_var, np.ndarray)
        self.assertIsInstance(components, np.ndarray)
        self.assertIsInstance(p_value, float)
        self.assertIsInstance(kmo_model, float)

        # Check if the dimensions of the transformed data are correct
        self.assertEqual(X_transformed.shape[0], X.shape[0])
        self.assertEqual(X_transformed.shape[1], n_components)

        # Check if explained variance ratio has the correct length
        self.assertEqual(len(exp_var), n_components)

        # Check if the shape of components matrix is correct
        self.assertEqual(components.shape[0], n_components)
        self.assertEqual(components.shape[1], X.shape[1])

        # Check if p-value and KMO model have valid values (not NaN)
        self.assertFalse(np.isnan(p_value))
        self.assertFalse(np.isnan(kmo_model))


if __name__ == '__main__':
    unittest.main()
