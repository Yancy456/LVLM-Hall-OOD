from typing import Literal, Optional
from .POPE import POPEDataset
from utils.prompt import Prompter
import argparse
from torch.utils.data import DataLoader
import os


def load_data(dataset_name: str, prompter: Prompter, annotation_path: str, data_folder: str,
              split: Literal['train', 'val', 'test'], batch_size: int = 1, category: Optional[str] = None) -> DataLoader:
    '''
    Load data from dataset 'dataset_name'.
    propter: Prompter used to construct prompts
    annotation_path: the path of annotation file
    data_folder:  the path of data folder
    split: choose from train, val or test set
    category: some dataset contains different category
    '''

    if dataset_name == 'POPE':
        data = POPEDataset(annotation_path,
                           data_folder, split, category).get_data()

    indices_to_keep = []
    for i in range(len(data)):  # check image existance
        if os.path.isfile(data[i]['img_path']):
            indices_to_keep.append(i)

    return data.select(indices_to_keep)
