import torch
from metric_utils import get_measures, print_measures
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


def svd_embed_score(embed_generated_wild, gt_label, begin_k, k_span, mean=1, svd=1, weight=0):
    '''search best k and layer'''
    embed_generated = embed_generated_wild  # the interal results of LLMs
    best_auroc_over_k = 0
    best_layer_over_k = 0
    best_scores_over_k = None
    best_projection_over_k = None

    for k in tqdm(range(begin_k, k_span)):  # search for best k
        best_auroc = 0
        best_layer = 0
        best_scores = None
        mean_recorded = None
        best_projection = None
        # iterate through layers to get projection
        for layer in range(len(embed_generated_wild[0])):
            if mean:
                mean_recorded = embed_generated[:, layer, :].mean(0)
                centered = embed_generated[:, layer, :] - mean_recorded
            else:
                centered = embed_generated[:, layer, :]

            if not svd:
                pca_model = PCA(n_components=k, whiten=False).fit(centered)
                projection = pca_model.components_.T
                mean_recorded = pca_model.mean_
                if weight:
                    projection = pca_model.singular_values_ * projection
            else:
                _, sin_value, V_p = torch.linalg.svd(
                    torch.from_numpy(centered).cuda())
                projection = V_p[:k, :].T.cpu().data.numpy()
                if weight:
                    projection = sin_value[:k] * projection  # get projection

            scores = np.mean(
                np.matmul(centered, projection), -1, keepdims=True)
            assert scores.shape[1] == 1
            scores = np.sqrt(np.sum(np.square(scores), axis=1))

            # not sure about whether true and false data the direction will point to,
            # so we test both. similar practices are in the representation engineering paper
            # https://arxiv.org/abs/2310.01405
            measures1 = get_measures(scores[gt_label == 1],
                                     # measure AUROC
                                     scores[gt_label == 0], plot=False)
            measures2 = get_measures(-scores[gt_label == 1],
                                     -scores[gt_label == 0], plot=False)

            if measures1[0] > measures2[0]:
                measures = measures1
                sign_layer = 1
            else:
                measures = measures2
                sign_layer = -1

            if measures[0] > best_auroc:
                best_auroc = measures[0]
                best_result = [100 * measures[2], 100 * measures[0]]
                best_layer = layer
                best_scores = sign_layer * scores
                best_projection = projection
                best_mean = mean_recorded
                best_sign = sign_layer
        print('k: ', k, 'best result: ', best_result, 'layer: ', best_layer,
              'mean: ', mean, 'svd: ', svd)

        if best_auroc > best_auroc_over_k:
            best_auroc_over_k = best_auroc
            best_result_over_k = best_result
            best_layer_over_k = best_layer
            best_k = k
            best_sign_over_k = best_sign
            best_scores_over_k = best_scores
            best_projection_over_k = best_projection
            best_mean_over_k = best_mean

    return {'k': best_k,
            'best_layer': best_layer_over_k,
            'best_auroc': best_auroc_over_k,
            'best_result': best_result_over_k,
            'best_scores': best_scores_over_k,
            'best_mean': best_mean_over_k,
            'best_sign': best_sign_over_k,
            'best_projection': best_projection_over_k}
