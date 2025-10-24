import unittest

import pandas as pd

from neurostatx.statistics.models import PHQ9Labeler


class TestFuzzyLabelers(unittest.TestCase):

    def test_fuzzy_labeler(self):
        # Test cases for the fuzzy_labeler function
        # We define score for 9 questions, but only Q1, Q2, Q6, and Q9 are
        # used.
        test_cases = [
            (3, 0, 0, 0, 0, 0, 0, 0, 0, "Not Depressed"),
            (0, 2, 0, 0, 0, 1, 0, 0, 0, "Not Depressed"),
            (1, 0, 0, 0, 0, 1, 0, 0, 1, "Not Depressed"),
            (2, 1, 1, 0, 0, 1, 0, 0, 1, "Mild"),
            (2, 2, 0, 0, 0, 0, 0, 0, 0, "Moderate"),
            (1, 3, 0, 0, 0, 1, 0, 0, 0, "Moderate"),
            (2, 1, 2, 3, 0, 2, 0, 0, 1, "Moderate"),
            (3, 1, 0, 0, 0, 1, 0, 0, 1, "Moderate"),
            (0, 2, 2, 2, 2, 2, 0, 3, 2, "Mod-Severe"),
            (2, 2, 0, 1, 1, 1, 3, 3, 1, "Moderate"),
            (3, 2, 0, 1, 1, 3, 3, 0, 0, "Severe"),
            (2, 2, 1, 1, 1, 2, 0, 0, 2, "Severe"),
            (3, 3, 1, 0, 0, 1, 0, 0, 1, "Mod-Severe"),
            (1, 3, 1, 1, 1, 2, 1, 1, 2, "Mod-Severe"),
            (2, 3, 2, 2, 2, 3, 2, 2, 1, "Mod-Severe"),
            (3, 2, 2, 2, 2, 2, 2, 2, 2, "Severe"),
            (2, 2, 0, 0, 0, 2, 0, 0, 3, "Severe"),
            (3, 3, 0, 1, 1, 3, 0, 0, 0, "Severe"),
            (3, 3, 0, 1, 1, 2, 3, 2, 2, "Severe"),
            (2, 3, 1, 1, 1, 3, 0, 0, 3, "Severe"),
            (3, 3, 0, 1, 1, 3, 0, 3, 3, "Severe")
        ]

        # Transform test cases into DataFrame and run the labeler
        test_cases_df = pd.DataFrame(test_cases,
                                     columns=['Q1', 'Q2', 'Q3', 'Q4', 'Q5',
                                              'Q6', 'Q7', 'Q8', 'Q9',
                                              'expected'])

        results = PHQ9Labeler().transform(test_cases_df.iloc[:, :-1])
        print(results)
        print(test_cases_df['expected'])
        self.assertTrue((results == test_cases_df['expected'].tolist()).all())
