import os

import numpy as np

from dataset.base import BaseDataset
from utils.func import read_jsonl
from typing import Literal, Optional
from utils.prompt import Prompter
import pandas as pd
from datasets import Dataset
from tqdm import tqdm


class POPEDataset():
    def __init__(self, annotation_path: str, data_folder: str):
        self.ann_path = annotation_path
        self.img_root = data_folder

    def get_data(self) -> list:
        ann = read_jsonl(self.ann_path)
        data_cat = [
            {
                "img_path": os.path.join(self.img_root, ins['image']),
                "question": f"{ins['text']}\nAnswer the question using a single word or phrase.",
                "label": 0 if ins['label'] == 'no' else 1,  # 1 yes; 0 no
                "question_id": ins["question_id"],
            }
            for ins in ann
        ]

        data = []
        for i in tqdm(range(len(data_cat))):  # check image existence
            if os.path.isfile(data_cat[i]['img_path']):
                data.append(data_cat[i])
        return data
