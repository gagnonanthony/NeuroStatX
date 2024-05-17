import unittest
import pandas as pd
import numpy as np

from neurostatx.statistics.harmonization import neuroCombat


class TestHarmonization(unittest.TestCase):

    def test_neuroCombat_noneb(self):
        # Define a dataset to harmonize.
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
                           'B': [5.0, 4.0, 3.0, 3.0, 2.0, 1.0],
                           'C': [1.0, 2.0, 3.0, 3.0, 4.0, 5.0]})

        # Define a dataset with batch information.
        covars = pd.DataFrame([1, 1, 1, 2, 2, 2], columns=['batch'])

        # Harmonize the dataset.
        out = neuroCombat(df.values.T,
                          covars=covars,
                          batch_col='batch',
                          eb=False,
                          ref_batch=1)

        expected_output = pd.DataFrame({'A': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                                        'B': [5.0, 4.0, 3.0, 5.0, 4.0, 3.0],
                                        'C': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]})

        # Compare the output with the expected output.
        np.testing.assert_allclose(out['data'],
                                   expected_output.values.T,
                                   atol=0.2)

    def test_neuroCombat_nonparam(self):
        # Define a dataset to harmonize.
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
                           'B': [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
                           'C': [1.0, 2.0, 3.0, 3.0, 4.0, 5.0]})

        # Define a dataset with batch information.
        covars = pd.DataFrame([1, 1, 1, 2, 2, 2], columns=['batch'])

        # Harmonize the dataset.
        out = neuroCombat(df.values.T,
                          covars=covars,
                          batch_col='batch',
                          eb=True,
                          parametric=False,
                          ref_batch=1)

        expected_output = pd.DataFrame({'A': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                                        'B': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                                        'C': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]})

        # Compare the output with the expected output.
        np.testing.assert_allclose(out['data'],
                                   expected_output.values.T,
                                   atol=0.2)

    def test_neuroCombat_param(self):
        # Define a dataset to harmonize.
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
                           'B': [5.0, 4.0, 3.0, 3.0, 2.0, 1.0],
                           'C': [1.0, 2.0, 3.0, 3.0, 4.0, 5.0]})

        # Define a dataset with batch information.
        covars = pd.DataFrame([1, 1, 1, 2, 2, 2], columns=['batch'])

        # Harmonize the dataset.
        out = neuroCombat(df.values.T,
                          covars=covars,
                          batch_col='batch',
                          eb=True,
                          ref_batch=1)

        expected_output = pd.DataFrame({'A': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                                        'B': [5.0, 4.0, 3.0, 5.0, 4.0, 3.0],
                                        'C': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]})

        # Compare the output with the expected output.
        np.testing.assert_allclose(out['data'],
                                   expected_output.values.T,
                                   atol=0.35)
