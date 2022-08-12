import unittest
from src.model_evaluator.evaluate_model import ModelEvaluator
import numpy as np
import numpy.testing as npt


class MyTestCase(unittest.TestCase):
    def test_computing_correct_metrics(self):
        eval = ModelEvaluator()
        actual = [0,0,1]
        pred = [0,0,1]
        cm, accuracy, f1 = eval.metrics_score(actual, pred)
        print(cm, accuracy, f1)
        self.assertEqual(accuracy, 1.0)  # add assertion here
        self.assertEqual(f1, 1)
        npt.assert_array_equal(np.array([[2, 0], [0, 1]], dtype=np.int64), cm)

    def test_incorrect_lengths_throw_exception(self):
        eval = ModelEvaluator()
        actual = [0, 1]
        pred = [0, 0, 1]
        with self.assertRaises(Exception)as context:
            eval.metrics_score(actual, pred)
            self.assertTrue(' input variables with inconsistent numbers' in context.exception)


if __name__ == '__main__':
    unittest.main()
