import unittest
from sklearn import datasets
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from neurostatx.statistics.models import plsr_cv


class TestPLSRFunctions(unittest.TestCase):

    def test_plsr_cv(self):
        # Generate synthetic data
        # Not optimal, but gives a certain quality control over the results.

        # Load the Diabetes dataset
        diabetes = datasets.load_diabetes()

        # Extract features (X) and target variable (y)
        X = diabetes.data
        y = diabetes.target

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=1234)

        # Initialize PLS model with the desired number of components
        n_components = 3
        pls_model = PLSRegression(n_components=n_components)

        # Fit the model on the training data
        pls_model.fit(X_train, y_train)

        # Predictions on the test set
        y_pred = pls_model.predict(X_test)

        # Evaluate the model performance
        r_squared = pls_model.score(X_test, y_test)
        mse_expected = mean_squared_error(y_test, y_pred)

        # Using the pls_cv function.
        plsr, mse, score_c, score_cv, rscore, mse_c, mse_cv = plsr_cv(
            X, y, nb_comp=3, splits=10, processes=4)

        # Validating score and mse are within a close range.
        self.assertAlmostEqual(score_c, r_squared, delta=0.05)
        self.assertAlmostEqual(mse_c, mse_expected, delta=100)


if __name__ == '__main__':
    unittest.main()
