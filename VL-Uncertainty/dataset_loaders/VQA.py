import os
import numpy as np

from dataset.base import BaseDataset
from utils.func import read_jsonl
from typing import Literal, Optional
from utils.prompt import Prompter
import pandas as pd
import json
from datasets import Dataset
from tqdm import tqdm


class VQADataset():
    def __init__(self, annotation_path: str, data_folder: str, split: Literal['train', 'val']):
        self.ann_path = annotation_path
        self.img_root = data_folder
        self.split = split
        if not any([self.split == x for x in ['train', 'val']]):
            raise ValueError(f'split {self.split} no supported')

    def get_data(self) -> list:
        data = []
        with open(self.ann_path, 'r') as file:
            ann = json.load(file)
        data_cat = [
            {
                "img_path": os.path.join(self.img_root, f"COCO_{self.split}2014_{ins['image_id']:012}.jpg"),
                "question": f"{ins['question']}\nAnswer the question using a single word or phrase.\n",
                "answers": ins['answers'],
                "question_id": ins["question_id"]
            }
            for ins in tqdm(ann)
        ]
        data += data_cat

        existing_data = []
        for ins in tqdm(data):
            if os.path.exists(ins['img_path']):
                existing_data.append(ins)
        print(f'existing files: {len(existing_data)}')
        return existing_data
