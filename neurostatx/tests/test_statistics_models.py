import unittest
from sklearn import datasets
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, permutation_test_score
from sklearn.metrics import mean_squared_error

from neurostatx.statistics.models import plsr_cv, permutation_testing


class TestPLSRFunctions(unittest.TestCase):

    def setUp(self) -> None:
        # Generate synthetic data
        # Not optimal, but gives a certain quality control over the results.

        # Load the Diabetes dataset
        diabetes = datasets.load_diabetes()

        # Extract features (X) and target variable (y)
        self.X = diabetes.data
        self.y = diabetes.target

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=0.2,
                                                            random_state=1234)

        # Initialize PLS model with the desired number of components
        n_components = 3
        self.pls_model = PLSRegression(n_components=n_components)

        # Fit the model on the training data
        self.pls_model.fit(X_train, y_train)

        # Predictions on the test set
        y_pred = self.pls_model.predict(X_test)

        # Evaluate the model performance
        self.r_squared = self.pls_model.score(X_test, y_test)
        self.mse_expected = mean_squared_error(y_test, y_pred)

        return super().setUp()

    def test_plsr_cv(self):
        # Using the pls_cv function.
        plsr, mse, score_c, score_cv, rscore, mse_c, mse_cv = plsr_cv(
            self.X, self.y, nb_comp=3, splits=10, processes=1)

        # Validating score and mse are within a close range.
        self.assertAlmostEqual(score_c, self.r_squared, delta=0.05)
        self.assertAlmostEqual(mse_c, self.mse_expected, delta=100)

    def test_permutation_testing(self):
        # Since this implementation is a modified version from
        # sklearn's permutation testing, we will assert the p-value obtained
        # from the permutation testing is identical.
        # Supply the precomputed PLSR model to the permutation testing function
        mod, sc, c, psc, spval, pc, cpval = permutation_testing(
            self.pls_model, self.X, self.y, nb_permutations=100, splits=10,
            processes=1)

        # Using the permutation testing function from sklearn
        sk_sc, sk_psc, sk_pval = permutation_test_score(
            self.pls_model, self.X, self.y, n_permutations=100,
            cv=10, n_jobs=1)

        # Assert the score and p-values are identical.
        self.assertEqual(sc, sk_sc)
        self.assertEqual(spval, sk_pval)


if __name__ == '__main__':
    unittest.main()
