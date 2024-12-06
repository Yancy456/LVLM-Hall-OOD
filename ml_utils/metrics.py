import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def auroc(scores, labels):
    return roc_auc_score(labels, scores)


def get_best_split_from_scores(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Calculate Youden's J statistic
    youdens_j = tpr - fpr
    # Find the index of the maximum J statistic
    best_index = np.argmax(youdens_j)
    best_threshold = thresholds[best_index]

    return best_threshold
