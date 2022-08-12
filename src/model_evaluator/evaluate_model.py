from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, accuracy_score, f1_score


class ModelEvaluator:

    def __init__(self):
        pass

    @classmethod
    def metrics_score(cls, actual, predicted):

        cm = confusion_matrix(actual, predicted)
        accuracy = accuracy_score(actual, predicted)
        f1 = f1_score(actual, predicted, average='weighted')
        return cm, accuracy, f1
