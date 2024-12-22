import random


def data_sampler(dataset, num_samples: int, shuffle: bool):
    '''Select some of the data, and shuffle data.

    '''
    if isinstance(dataset, list):
        if shuffle:
            random.shuffle(dataset)
        return dataset[:num_samples]
    else:  # huggingface Dataset
        dataset.shuffle()
        return dataset.select(range(num_samples))
