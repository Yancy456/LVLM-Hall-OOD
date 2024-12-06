import random

def data_sampler(data:list,num_samples:int,shuffle:bool=False):
    '''Select some of the data, and shuffle data.
    
    '''
    if shuffle:
        random.shuffle(data)
    return data[:num_samples]