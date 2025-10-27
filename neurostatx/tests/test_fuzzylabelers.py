import unittest

import pandas as pd

from neurostatx.statistics.models import PHQ9Labeler, GAD7Labeler


class TestFuzzyLabelers(unittest.TestCase):

    def test_fuzzy_labeler(self):
        # Test cases for the fuzzy_labeler function
        # We define score for 9 questions, but only Q1, Q2, Q6, and Q9 are
        # used.
        test_cases = [
            (3, 0, 0, 0, 0, 0, 0, 0, 0, "Subthreshold"),
            (0, 2, 0, 0, 0, 1, 0, 0, 0, "Subthreshold"),
            (1, 0, 0, 0, 0, 1, 0, 0, 1, "Subthreshold"),
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
        results_fit = PHQ9Labeler().fit(test_cases_df.iloc[:, :-1])
        self.assertTrue((results == test_cases_df['expected'].tolist()).all())
        self.assertTrue(
            (results_fit == test_cases_df['expected'].tolist()).all())

    def test_fuzzy_labeler_gad7(self):
        # Test case for the GAD-7 labeler.
        # We define mock scores for 7 questions, trying to cover all cases.
        test_cases = [
            (0, 0, 0, 0, 0, 0, 0, "Subthreshold"),
            (1, 0, 0, 0, 0, 0, 0, "Subthreshold"),
            (0, 2, 3, 0, 0, 0, 0, "Subthreshold"),
            (1, 1, 1, 1, 1, 2, 0, "Mild"),
            (2, 2, 2, 0, 0, 0, 0, "Mild"),
            (3, 3, 2, 0, 0, 0, 0, "Mild"),
            (2, 2, 1, 1, 1, 1, 1, "Mild"),
            (3, 3, 3, 0, 0, 0, 0, "Moderate"),
            (3, 0, 2, 2, 1, 1, 0, "Moderate"),
            (2, 2, 2, 1, 1, 1, 0, "Moderate"),
            (2, 2, 2, 2, 1, 0, 0, "Moderate"),
            (3, 2, 1, 1, 1, 1, 1, "Mild"),
            (2, 2, 2, 1, 1, 1, 1, "Mild"),
            (3, 3, 3, 0, 0, 0, 1, "Moderate"),
            (3, 3, 1, 1, 0, 1, 1, "Moderate"),
            (2, 2, 2, 2, 1, 1, 0, "Moderate"),
            (2, 2, 2, 3, 1, 0, 0, "Moderate"),
            (3, 2, 2, 1, 1, 1, 1, "Mild"),
            (2, 2, 2, 2, 1, 1, 1, "Mild"),
            (3, 3, 3, 1, 1, 0, 0, "Moderate"),
            (2, 2, 2, 2, 2, 1, 0, "Moderate"),
            (3, 3, 2, 2, 1, 0, 0, "Moderate"),
            (3, 3, 3, 3, 0, 0, 0, "Moderate"),
            (2, 2, 2, 2, 2, 2, 0, "Moderate"),
            (3, 3, 3, 3, 1, 0, 0, "Moderate"),
            (3, 3, 3, 1, 1, 1, 1, "Moderate"),
            (3, 3, 2, 2, 1, 1, 1, "Moderate"),
            (3, 2, 2, 2, 2, 2, 0, "Moderate"),
            (2, 2, 2, 2, 2, 2, 1, "Moderate"),
            (3, 3, 3, 2, 2, 0, 0, "Severe"),
            (3, 3, 2, 2, 2, 1, 0, "Severe"),
            (3, 3, 3, 3, 1, 0, 1, "Moderate"),
            (3, 3, 3, 2, 1, 1, 1, "Moderate"),
            (2, 2, 2, 2, 2, 2, 2, "Moderate"),
            (3, 3, 3, 3, 2, 0, 0, "Severe"),
            (3, 3, 3, 2, 2, 1, 0, "Severe"),
            (3, 3, 2, 2, 2, 2, 0, "Severe"),
            (3, 3, 2, 2, 2, 1, 1, "Severe"),
            (3, 2, 2, 2, 2, 2, 1, "Severe"),
            (3, 3, 2, 2, 2, 2, 1, "Severe"),
            (3, 2, 2, 2, 2, 2, 2, "Severe"),
            (3, 3, 3, 2, 1, 1, 2, "Severe"),
            (3, 3, 3, 3, 3, 0, 0, "Severe"),
            (3, 3, 3, 3, 2, 1, 0, "Severe"),
            (3, 3, 3, 2, 2, 2, 0, "Severe"),
            (3, 3, 3, 3, 3, 1, 0, "Severe"),
            (3, 3, 3, 2, 2, 2, 2, "Severe"),
            (3, 3, 3, 3, 2, 2, 2, "Severe"),
            (3, 3, 3, 3, 3, 2, 2, "Severe"),
            (3, 3, 3, 3, 3, 3, 2, "Severe"),
            (3, 3, 3, 3, 3, 3, 3, "Severe")
        ]

        # Transform test cases into DataFrame and run the labeler
        test_cases_df = pd.DataFrame(test_cases,
                                     columns=['Q1', 'Q2', 'Q3', 'Q4', 'Q5',
                                              'Q6', 'Q7', 'expected'])
        results = GAD7Labeler().transform(
            test_cases_df.iloc[:, :-1])
        results_fit = GAD7Labeler().fit(
            test_cases_df.iloc[:, :-1])
        self.assertTrue((results == test_cases_df['expected'].tolist()).all())
        self.assertTrue(
            (results_fit == test_cases_df['expected'].tolist()).all())

    def test_assess_ValueError_PHQ9_fit(self):
        # Test that invalid inputs raise ValueError
        invalid_data = pd.DataFrame({
            'Q1': [4, -1, 2],
            'Q2': [0, 1, 5],
            'Q3': [1, 2, 3],
            'Q4': [0, 0, 0],
            'Q5': [0, 0, 0],
            'Q6': [0, 0, 0],
            'Q7': [0, 0, 0],
            'Q8': [0, 0, 0],
            'Q9': [0, 0, 0]
        })

        with self.assertRaises(ValueError):
            PHQ9Labeler().fit(invalid_data.iloc[:, :-3])

    def test_assess_ValueError_PHQ9_transform(self):
        # Test that invalid inputs raise ValueError
        invalid_data = pd.DataFrame({
            'Q1': [4, -1, 2],
            'Q2': [0, 1, 5],
            'Q3': [1, 2, 3],
            'Q4': [0, 0, 0],
            'Q5': [0, 0, 0],
            'Q6': [0, 0, 0],
            'Q7': [0, 0, 0],
            'Q8': [0, 0, 0],
            'Q9': [0, 0, 0]
        })

        with self.assertRaises(ValueError):
            PHQ9Labeler().transform(invalid_data.iloc[:, :-3])

    def test_assess_ValueError_GAD7_fit(self):
        # Test that invalid inputs raise ValueError
        invalid_data = pd.DataFrame({
            'Q1': [4, -1, 2],
            'Q2': [0, 1, 5],
            'Q3': [1, 2, 3],
            'Q4': [0, 0, 0],
            'Q5': [0, 0, 0],
            'Q6': [0, 0, 0],
            'Q7': [0, 0, 0],
            'Q8': [0, 0, 0],
            'Q9': [0, 0, 0]
        })

        with self.assertRaises(ValueError):
            GAD7Labeler().fit(invalid_data.iloc[:, :-3])

    def test_assess_ValueError_GAD7_transform(self):
        # Test that invalid inputs raise ValueError
        invalid_data = pd.DataFrame({
            'Q1': [4, -1, 2],
            'Q2': [0, 1, 5],
            'Q3': [1, 2, 3],
            'Q4': [0, 0, 0],
            'Q5': [0, 0, 0],
            'Q6': [0, 0, 0],
            'Q7': [0, 0, 0],
            'Q8': [0, 0, 0],
            'Q9': [0, 0, 0]
        })

        with self.assertRaises(ValueError):
            GAD7Labeler().transform(invalid_data.iloc[:, :-3])
