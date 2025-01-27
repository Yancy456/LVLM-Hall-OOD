import torch
from tqdm import tqdm
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import numpy as np
from typing import Literal
from scipy.linalg import svd


class PCAKernel:
    '''A class that uses PCA to score data'''

    def __init__(self, X, n_components, kernel: Literal['poly', 'rbf', 'linear'] = 'linear') -> None:
        # X.shape= (num_samples,dimensions_of_hidden_states)

        pca_model = KernelPCA(n_components=n_components,
                              kernel=kernel, n_jobs=-1).fit(X)

        self.X = X
        self.pca = pca_model

    def get_score(self, X):
        X = self.pca.transform(X)
        scores = np.sqrt(
            np.sum(np.square(X), -1))
        return scores

    def get_best_split(self, scores, y):
        '''get best split from scores'''
        fpr, tpr, thresholds = roc_curve(y, scores)

        # Calculate Youden's J statistic
        youdens_j = tpr - fpr
        # Find the index of the maximum J statistic
        best_index = np.argmax(youdens_j)
        best_threshold = thresholds[best_index]

        return best_threshold


class PCALinear:
    '''A class that uses PCA to score data'''

    def __init__(self, X, n_components) -> None:
        # X.shape= (num_samples,dimensions_of_hidden_states)

        pca_model = PCA(n_components=n_components, whiten=False).fit(X)
        projections = pca_model.singular_values_*pca_model.components_.T
        mean_recorded = pca_model.mean_

        self.X = X
        self.projections = projections

    def get_score(self, X):
        scores = np.mean(
            np.matmul(X, self.projections), -1, keepdims=True)
        assert scores.shape[1] == 1
        scores = np.sqrt(np.sum(np.square(scores), axis=1))
        return scores  # scores.shape=(num_samples)

    def get_best_split(self, scores, y):
        '''get best split from scores'''
        fpr, tpr, thresholds = roc_curve(y, scores)

        # Calculate Youden's J statistic
        youdens_j = tpr - fpr
        # Find the index of the maximum J statistic
        best_index = np.argmax(youdens_j)
        best_threshold = thresholds[best_index]

        return best_threshold


class KernelPCA:
    def __init__(self, X, n_components, M, method: Literal['CoRP', 'CoP', 'origin'] = 'CoRP', gamma=1) -> None:
        '''
        X.shape=[n,m], n: number of samples, m: number of hidden states
        '''

        X = self._kernel_projection(X, M, gamma, method)
        pca_model = PCA(n_components=n_components, whiten=False).fit(X)
        projections = pca_model.singular_values_*pca_model.components_.T
        mean_recorded = pca_model.mean_

        self.X = X
        self.projections = projections

    def get_score(self, X):
        scores = np.mean(
            np.matmul(X, self.projections), -1, keepdims=True)
        assert scores.shape[1] == 1
        scores = np.sqrt(np.sum(np.square(scores), axis=1))
        return scores  # scores.shape=(num_samples)

    def get_best_split(self, scores, y):
        '''get best split from scores'''
        fpr, tpr, thresholds = roc_curve(y, scores)

        # Calculate Youden's J statistic
        youdens_j = tpr - fpr
        # Find the index of the maximum J statistic
        best_index = np.argmax(youdens_j)
        best_threshold = thresholds[best_index]

        return best_threshold

    def _kernel_projection(self, X, M: int, gamma: float, method: Literal['CoRP', 'CoP', 'origin'] = 'CoRP'):
        '''
        X.shape=[n,m], n: number of samples, m: number of features
        '''
        if method == 'origin':
            return X

        def normalizer(x): return x / (np.linalg.norm(x,
                                                      ord=2, axis=-1, keepdims=True) + 1e-10)
        if method == 'CoRP':
            X = normalizer(X)
            m = X.shape[1]
            # generate M i.i.d. samples from p(w)
            w = np.sqrt(2*gamma)*np.random.normal(size=(M, m))
            u = 2 * np.pi * np.random.rand(M)

            X = np.sqrt(2/M)*np.cos((X@w.T + u[np.newaxis, :]))
            return X
        elif method == 'CoP':
            X = normalizer(X)
            return X
        else:
            raise ValueError('method unsupported')


# class KernelPCA:
#    def __init__(self, X, M, gamma, method: Literal['CoRP', 'CoP', 'origin'] = 'CoRP') -> None:
#        '''
#        X.shape=[n,m], n: number of samples, m: number of hidden states
#        '''

#        X = self._kernel_projection(X, M, gamma, method)
#        mu = X.mean(axis=0)
#        X = X-mu
#        K = X.T@X
#        u, s, _ = svd(K)

#        self.X = X
#        self.u = u
#        self.s = s

#    def get_score(self, X, n_components):
#        u_q = self.u[:, :n_components]  # u_q.shape=[m,q]
#        reconstruct = u_q.dot(u_q.T).dot(X.T).T
#        scores = - np.linalg.norm(X-reconstruct, ord=2, axis=1)
#        return scores

#    def get_score_hall(self, X, n_components):
#        u_q = self.u[:, :n_components]  # u_q.shape=[m,q]
#        singulars = self.s[:n_components]
#        projections = singulars*u_q  # shape=[m,q]
#        scores = np.mean(
#            np.square(np.matmul(X, projections)), -1)

#        return scores  # scores.shape=(num_samples)

#    def get_best_split(self, scores, y, n_components):
#        '''get best split from scores'''
#        fpr, tpr, thresholds = roc_curve(y, scores)

#        # Calculate Youden's J statistic
#        youdens_j = tpr - fpr
#        # Find the index of the maximum J statistic
#        best_index = np.argmax(youdens_j)
#        best_threshold = thresholds[best_index]

#        return best_threshold

#    def get_acc(self, split, y):
#        scores = self.get_score()
#        preds = (scores > split)
#        return accuracy_score(y, preds)

#    def _kernel_projection(self, X, M: int, gamma: float, method: Literal['CoRP', 'CoP', 'origin'] = 'CoRP'):
#        '''
#        X.shape=[n,m], n: number of samples, m: number of features
#        '''
#        if method == 'origin':
#            return X

#        def normalizer(x): return x / (np.linalg.norm(x,
#                                                      ord=2, axis=-1, keepdims=True) + 1e-10)
#        X = normalizer(X)

#        if method == 'CoRP':
#            m = X.shape[1]
#            # generate M i.i.d. samples from p(w)
#            w = np.sqrt(2*gamma)*np.random.normal(size=(M, m))
#            u = 2 * np.pi * np.random.rand(M)

#            X = np.sqrt(2/M)*np.cos((X@w.T + u[np.newaxis, :]))
#        elif method == 'CoP':
#            return X
#        else:
#            raise ValueError('method unsupported')
#        return X
