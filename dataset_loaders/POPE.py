import os

import numpy as np

from dataset.base import BaseDataset
from utils.func import read_jsonl
from typing import Literal, Optional
from utils.prompt import Prompter
import pandas as pd
from datasets import Dataset


class POPEDataset():
    def __init__(self, annotation_path: str, data_folder: str,
                 split: Literal['train', 'val', 'test'], category: Literal["adversarial", "popular", "random"]):

        self.ann_path = annotation_path
        self.img_root = data_folder
        self.split = split
        self.category = category
        if category not in ["adversarial", "popular", "random"]:  # check category value
            raise ValueError(f'No such {category} in POPE dataset!')

    def get_data(self) -> Dataset:
        data = []
        ann = read_jsonl(self.ann_path)
        data_cat = [
            {
                "img_path": os.path.join(self.img_root, ins['image']),
                "question": f"{ins['text']}\nAnswer the question using a single word or phrase.",
                "label": 0 if ins['label'] == 'no' else 1,  # 1 yes; 0 no
                "question_id": ins["question_id"],
                "category": self.category
            }
            for ins in ann
        ]

        data += data_cat
        df = pd.DataFrame(data)
        return Dataset.from_pandas(df)
