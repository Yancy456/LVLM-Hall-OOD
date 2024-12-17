from typing import Literal, Optional
from .POPE import POPEDataset
from .MMSafety import MMSafetyBench
from .ScienceQA import ScienceQA
from utils.prompt import Prompter
import argparse
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HfDataset
from PIL import Image
import pandas as pd
from torchvision import transforms


def load_data(dataset_name: str, prompter: Prompter,  data_folder: str,
              split: Literal['train', 'val', 'test'],  annotation_path: Optional[str] = None, category: Optional[str] = None):
    '''
    Load data from dataset 'dataset_name'.
    prompter: Prompter used to construct prompts
    annotation_path: the path of annotation file
    data_folder:  the path of data folder
    split: choose from train, val or test set
    category: some dataset contains different category
    '''

    if dataset_name == 'POPE':
        data = POPEDataset(annotation_path,
                           data_folder, split, category).get_data()
    elif dataset_name == 'MMSafety':
        data = MMSafetyBench(prompter, data_folder, split).get_data()
    elif dataset_name == 'ScienceQA':
        data = ScienceQA(split).get_data()
    else:
        raise ValueError(f'No such dataset {dataset_name}')

    # indices_to_keep = []
    # for i in range(len(data)):  # check image existence
    #    if (not isinstance(data[i]['img_path'], str)) or os.path.isfile(data[i]['img_path']):
    #        indices_to_keep.append(i)
    return data
    # return data.select(indices_to_keep)


# Define the custom dataset class
class ImageDataset(Dataset):
    def __init__(self, dataset: HfDataset):
        self.dataset = dataset
        self.img_size = (400, 400)
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()  # Convert PIL image to PyTorch tensor
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if 'img_path' in item:
            image = Image.open(item['img_path'])
        else:
            image = item['img']

        if item['img'] == None:
            # create black image if no image input
            image = Image.new("RGB", self.img_size, color=(255, 255, 255))

        item['img'] = self.transform(image)

        return item
