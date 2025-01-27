import os
import numpy as np
from typing import Literal, Optional
import json
from tqdm import tqdm


class ChartVQADataset():
    def __init__(self, annotation_path: str, data_folder: str):
        self.ann_path = annotation_path
        self.img_root = data_folder
        self.data = self.get_data()

    def get_data(self) -> list:
        data = []
        with open(self.ann_path, 'r') as file:
            ann = json.load(file)
        data_cat = [
            {
                "img": os.path.join(self.img_root, f"{ins['imgname']}"),
                "question": f"{ins['query']}\nAnswer the question using a single word or digit number.\n",
                "gt_ans": ins['label'],
                "question_id": ins["imgname"]
            }
            for ins in tqdm(ann)
        ]
        data += data_cat

        return data

    def obtain_size(self):
        return len(self.data)

    def retrieve(self, idx):
        row = self.data[idx]
        row['idx'] = idx
        return row
