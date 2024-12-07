import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import numpy as np


class PCADiscriminator:
    '''A class that uses PCA to score data'''

    def __init__(self, n_components, X) -> None:
        # X.shape= (num_samples,dimensions_of_hidden_states)

        pca_model = PCA(n_components=n_components, whiten=False).fit(X)
        projections = pca_model.singular_values_*pca_model.components_.T
        mean_recorded = pca_model.mean_

        self.X = X
        self.projections = projections

    def get_score(self):
        scores = np.mean(
            np.matmul(self.X, self.projections), -1, keepdims=True)
        assert scores.shape[1] == 1
        scores = np.sqrt(np.sum(np.square(scores), axis=1))
        return scores  # scores.shape=(num_samples)

    def get_best_split(self, y):
        '''get best split from scores'''
        scores = self.get_score()
        fpr, tpr, thresholds = roc_curve(y, scores)

        # Calculate Youden's J statistic
        youdens_j = tpr - fpr
        # Find the index of the maximum J statistic
        best_index = np.argmax(youdens_j)
        best_threshold = thresholds[best_index]

        return best_threshold

    def get_acc(self, split, y):
        scores = self.get_score()
        preds = (scores > split)
        return accuracy_score(y, preds)
