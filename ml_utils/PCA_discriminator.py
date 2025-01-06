import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import numpy as np
from typing import Literal
from scipy.linalg import svd


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


class KernelPCA:
    def __init__(self, X, M, gamma, method: Literal['CoRP', 'CoP'] = 'CoRP') -> None:
        '''
        X.shape=[n,m], n: number of samples, m: number of hidden states
        '''

        X = self._kernel_projection(X, M, gamma, method)
        mu = X.mean(axis=0)
        X = X-mu
        K = X@X.T
        u, s, _ = svd(K)

        self.X = X
        self.u = u

    def get_score(self, X, n_components):
        u_q = self.u[:, :n_components]
        reconstruct = u_q.dot(u_q.T).dot(X)
        scores = - np.linalg.norm(X-reconstruct, ord=2, axis=1)
        return scores

    def get_best_split(self, X, y, n_components):
        '''get best split from scores'''
        scores = self.get_score(X, n_components)
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

    def _kernel_projection(self, X, M: int, gamma: float, method: Literal['CoRP', 'CoP'] = 'CoRP'):
        '''
        X.shape=[n,m], n: number of samples, m: number of features
        '''
        def normalizer(x): return x / (np.linalg.norm(x,
                                                      ord=2, axis=-1, keepdims=True) + 1e-10)
        X = normalizer(X)
        if method == 'CoRP':
            m = X.shape[1]
            # generate M i.i.d. samples from p(w)
            w = np.sqrt(2*gamma)*np.random.normal(size=(M, m))
            u = 2 * np.pi * np.random.rand(M)

            X = np.sqrt(2/M)*np.cos((X@w.T + u[np.newaxis, :]))
        elif method == 'CoP':
            return X
        else:
            raise ValueError('method unsupported')
        return X
