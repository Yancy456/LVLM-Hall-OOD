import os
import numpy as np

from typing import Literal, Optional
import pandas as pd
from datasets import load_dataset
from datasets import Dataset
from tqdm import tqdm
import json


class DocVQADataset():
    def __init__(self, annotation_path: str, data_folder: str):
        self.ann_path = annotation_path
        self.img_root = data_folder

    def get_data(self) -> list:
        with open(self.ann_path, 'r') as file:
            ann = json.load(file)
        data = [
            {
                "img_path": os.path.join(self.img_root, f"{ins['image']}"),
                "question": f"{ins['question']}\nAnswer the question using a single word or digit number.\n",
                "answer": ins['answers']
            }
            for ins in tqdm(ann['data'])
        ]
        return data
