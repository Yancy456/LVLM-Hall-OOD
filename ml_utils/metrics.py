import numpy as np
from sklearn.metrics import precision_recall_curve, auc


def auc_pr(y, scores):
    # Example: Assume y_test and y_score are your true labels and predicted scores

    # Calculate Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y, scores)

    return auc(recall, precision)
