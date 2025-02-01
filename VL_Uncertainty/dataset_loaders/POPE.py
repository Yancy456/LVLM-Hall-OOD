import os

import numpy as np
from typing import Literal, Optional
import json
from tqdm import tqdm
from utils.func import read_jsonl


class POPEDataset():
    def __init__(self, annotation_path: str, data_folder: str):

        self.ann_path = annotation_path
        self.img_root = data_folder
        self.data = self.get_data()

    def get_data(self) -> list:
        ann = read_jsonl(self.ann_path)
        data_cat = [
            {
                "img": os.path.join(self.img_root, ins['image']),
                "question": f"{ins['text']}\nAnswer the question using a single word or phrase.",
                "gt_ans": 0 if ins['label'] == 'no' else 1,  # 1 yes; 0 no
                "question_id": ins["question_id"]
            }
            for ins in ann
        ]

        data = []
        for i in tqdm(range(len(data_cat))):  # check image existence
            if os.path.isfile(data_cat[i]['img']):
                data.append(data_cat[i])
        return data

    def obtain_size(self):
        return len(self.data)

    def retrieve(self, idx):
        row = self.data[idx]
        row['idx'] = idx
        return row
