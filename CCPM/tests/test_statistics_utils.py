import numpy as np
import pandas as pd
import unittest

from CCPM.statistics.utils import KNNimputation


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


if __name__ == '__main__':
    unittest.main()
