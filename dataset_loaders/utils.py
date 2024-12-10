import random
from datasets import Dataset


def data_sampler(dataset: Dataset, num_samples: int, shuffle: bool = False):
    '''Select some of the data, and shuffle data.

    '''
    if shuffle:
        dataset.shuffle()
    return dataset.select([0, num_samples])
