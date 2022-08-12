import unittest
import pandas as pd
from src.models.ml_logistic_regression import MLModelLogisticRegression
import pandas.testing as pdt
import numpy.testing as npt


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.model = MLModelLogisticRegression()

    def test_training_correct(self):
        data = pd.read_csv('test/test_data/test_valid_teams.csv')
        X_train = data[['Day', 'Venue', 'Poss']]
        y_train = data[['Result']]

        self.model.train(X_train, y_train)
        expected_params = {'C': 1.0,
                           'class_weight': None,
                            'dual': False,
                            'fit_intercept': True,
                            'intercept_scaling': 1,
                            'l1_ratio': None,
                            'max_iter': 100,
                            'multi_class': 'ovr',
                            'n_jobs': None,
                            'penalty': 'l2',
                            'random_state': 10,
                            'solver': 'lbfgs',
                            'tol': 0.0001,
                            'verbose': 0,
                            'warm_start': False}
        self.assertEqual(self.model.clf.get_params(), expected_params)

    def test_pred_transform(self,):
        data = pd.read_csv('test/test_data/test_valid_teams.csv')
        X_train = data[['Day', 'Venue', 'Poss']]
        expected_result = pd.DataFrame.from_records([
            [1.404563,0.0,1.0,0.0,0.0,1.0],
            [-0.845118,0.0,0.0,1.0,1.0,0.0],
            [-0.559445,1.0,0.0,0.0,0.0,1.0]], columns=['Poss','Mon','Tue','Wed','Away','Home'])
        result = self.model._fit_preprocessor(X_train)
        pdt.assert_frame_equal(result, expected_result)
        npt.assert_almost_equal(result['Poss'].mean(), 0)
        # as dataset is small standard deviation may not be exactly 1.
        # npt.assert_almost_equal(result['Poss'].std(), 1)


if __name__ == '__main__':
    unittest.main()
