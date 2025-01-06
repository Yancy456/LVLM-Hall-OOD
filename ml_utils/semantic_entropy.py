import numpy as np
from matplotlib import pyplot as plt
import torch


# Best split for SE binarization.
def best_split(entropy: np.array, label="Dx"):
    """
    Identify best split for minimizing reconstruction error via low and high SE mean estimates,
    as discussed in Section 4. Binarization of paper (ArXiv: 2406.15927)
    """
    ents = entropy
    splits = np.linspace(1e-10, ents.max(), 100)
    split_mses = []
    for split in splits:
        low_idxs, high_idxs = ents < split, ents >= split

        low_mean = np.mean(ents[low_idxs])
        high_mean = np.mean(ents[high_idxs])

        mse = np.sum((ents[low_idxs] - low_mean)**2) + \
            np.sum((ents[high_idxs] - high_mean)**2)
        mse = np.sum(mse)

        split_mses.append(mse)

    split_mses = np.array(split_mses)

    # plt.plot(splits, split_mses, label=label)
    return splits[np.argmin(split_mses)]


# 0.0 means even splits for normalized entropy scores
def binarize_entropy(entropy, thres=0.0):
    """Binarize entropy scores into 0s and 1s"""
    binary_entropy = np.full_like(entropy, -1, dtype=np.float32)
    binary_entropy[entropy <= thres] = 1
    binary_entropy[entropy > thres] = 0

    return binary_entropy
