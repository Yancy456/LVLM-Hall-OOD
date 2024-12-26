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
from .VQA import VQADataset
from dataset_loaders.utils import data_sampler
from .VizWiz import VizWizDataset
from .AOKVQA import AOKVQADataset


def load_data(dataset_name: str, args):
    '''
    Load data from dataset 'dataset_name'.
    prompter: Prompter used to construct prompts
    annotation_path: the path of annotation file
    data_folder:  the path of data folder
    split: choose from train, val or test set
    category: some dataset contains different category
    '''

    if dataset_name == 'POPE':
        data = POPEDataset(args.annotation_path,
                           args.data_folder, args.split, args.category).get_data()
    elif dataset_name == 'MMSafety':
        data = MMSafetyBench(
            args.prompter, args.data_folder, args.split).get_data()
    elif dataset_name == 'ScienceQA':
        data = ScienceQA(args.split).get_data()
    elif dataset_name == 'VQA':
        data = VQADataset(args.annotation_path,
                          args.data_folder, args.split).get_data()
    elif dataset_name == 'VizWiz':
        data = VizWizDataset(args.annotation_path,
                             args.data_folder).get_data()
    elif dataset_name == 'AOKVQA':
        data = AOKVQADataset(args.annotation_path,
                             args.data_folder).get_data()
    else:
        raise ValueError(f'No such dataset {dataset_name}')

    return data


# Define the custom dataset class
class ImageDataset(Dataset):
    def __init__(self, dataset: HfDataset, image_shape: Optional[list[int]] = None):
        self.dataset = dataset

        # self.img_size = image_shape
        self.img_size = image_shape
        if self.img_size == None:
            self.transform = transforms.Compose([
                transforms.ToTensor()  # Convert PIL image to PyTorch tensor
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor()  # Convert PIL image to PyTorch tensor
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if 'img_path' in item:
            item['img'] = Image.open(item['img_path']).convert('RGB')

        if not 'img' in item or item['img'] == None:
            # create black image if no image input
            item['img'] = Image.new("RGB", (500, 500), color=(255, 255, 255))
        item['img'] = self.transform(item['img'])

        return item
