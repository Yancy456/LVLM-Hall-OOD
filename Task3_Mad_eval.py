# %%

from linear_probe import get_linear_acc
from metric_utils import get_measures, print_measures
from sklearn.decomposition import PCA
from ML_tools.PCA_discriminator import PCADiscriminator, svd_embed_score
import os
import json

import torch
import numpy as np
from glob import glob
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression

from utils.func import read_data, softmax
from utils.metric import evaluate
from utils.store_data import ReadData
import pandas as pd
model_name = "LLaVA-7B"
prompt = "mq"

# %%
train_reader = ReadData('./output/mad_train_save')
val_reader = ReadData('./output/mad_val_save')
trainset = train_reader.read_all()
valset = val_reader.read_all()
trainset, valset


X_train = np.array([ins['hidden_states'] for ins in trainset]).squeeze()
y_train = np.array([ins['label'] for ins in trainset]).squeeze()

X_val = np.array([ins['hidden_states'] for ins in valset]).squeeze()
y_val = np.array([ins['label'] for ins in valset]).squeeze()

val_data = valset

X_train.shape

# %%

returned_results = svd_embed_score(X_train, y_train,
                                   begin_k=1, k_span=11, mean=0, svd=0)


# discriminator=PCADiscriminator(k=6,X=X_test[:,5,:,:]) # select_layer


# %%

pca_model = PCA(n_components=returned_results['k'], whiten=False).fit(
    X_train[:, returned_results['best_layer'], :])
projection = pca_model.components_.T

scores = np.mean(np.matmul(
    X_train[:, returned_results['best_layer'], :], projection), -1, keepdims=True)
assert scores.shape[1] == 1
best_scores = np.sqrt(np.sum(np.square(scores), axis=1)
                      ) * returned_results['best_sign']

# direct projection
feat_indices_test = []

test_scores = np.mean(np.matmul(X_train[:, returned_results['best_layer'], :],
                                projection), -1, keepdims=True)

assert test_scores.shape[1] == 1
test_scores = np.sqrt(np.sum(np.square(test_scores), axis=1))

measures = get_measures(returned_results['best_sign'] * test_scores[y_train == 1],
                        returned_results['best_sign'] * test_scores[y_train == 0], plot=False)
print_measures(measures[0], measures[1], measures[2], 'direct-projection')

# %%

'''train linear classifier using labels obtained from SVD on test set'''

best_layer = None
best_clf = None
y_test_scores = None

thresholds = np.linspace(0, 1, num=20)[1:-1]

# graid search
auroc_over_thres = []
for thres_wild in thresholds:
    best_auroc = 0
    best_auroc_acc = 0
    for layer in range(X_train.shape[1]):
        thres_wild_score = np.sort(best_scores)[
            int(len(best_scores) * thres_wild)]
        true_wild = X_train[:, layer,
                            :][best_scores > thres_wild_score]
        false_wild = X_train[:, layer,
                             :][best_scores <= thres_wild_score]

        embed_train = np.concatenate(
            [true_wild, false_wild], 0).astype(np.float16)
        label_train = np.concatenate([np.ones(len(true_wild)),
                                      np.zeros(len(false_wild))], 0).astype(np.float16)

        # gt training, saplma
        # embed_train = embed_generated_wild[:,layer,:]
        # label_train = gt_label_wild
        # gt training, saplma

        best_acc, final_acc, (
            clf, best_state, best_preds, preds, labels_val), losses_train = get_linear_acc(
            embed_train,
            label_train,
            embed_train,
            label_train,
            2, epochs=50,
            print_ret=True,
            batch_size=512,
            cosine=True,
            nonlinear=True,
            learning_rate=0.05,
            weight_decay=0.0003)

        clf.eval()
        output = clf(torch.from_numpy(
            X_val[:, layer, :]).float().cuda())
        pca_wild_score_binary_cls = torch.sigmoid(output)

        pca_wild_score_binary_cls = pca_wild_score_binary_cls.cpu().data.numpy()

        if np.isnan(pca_wild_score_binary_cls).sum() > 0:
            breakpoint()
        measures = get_measures(pca_wild_score_binary_cls[y_val == 1],
                                pca_wild_score_binary_cls[y_val == 0], plot=False)

        if measures[0] > best_auroc:
            best_auroc = measures[0]
            best_result = [100 * measures[0]]
            best_layer = layer
            best_auroc_acc = best_acc
            y_test_scores = pca_wild_score_binary_cls
            best_clf = clf

    auroc_over_thres.append(best_auroc)
    print('thres: ', thres_wild, 'best result: ',
          best_result, 'best_layer: ', best_layer, f'acc:{best_auroc_acc}')


# %%

clf.eval()
output = clf(torch.from_numpy(
    X_val[:, best_layer, :]).float().cuda())
pca_wild_score_binary_cls = torch.sigmoid(
    output).detach().cpu().numpy().squeeze()

# %%
_, _, _, _ = evaluate(y_val, pca_wild_score_binary_cls, show=True)
# %%
