from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, accuracy_score, f1_score


class ModelEvaluator:
    """
    This class take the prediction and evaluates the model predictions
    """
    def __init__(self):
        pass

    @classmethod
    def metrics_score(cls, actual, predicted):
         """
        Takes the actual data and predictions and return the evaluation
        Param actual: Takes the data frame
        Param predicted: the prediction as input
        Return: Confusion Matrix, accuracy and F1 results
        """

        cm = confusion_matrix(actual, predicted)
        accuracy = accuracy_score(actual, predicted)
        f1 = f1_score(actual, predicted, average='weighted')
        return cm, accuracy, f1
