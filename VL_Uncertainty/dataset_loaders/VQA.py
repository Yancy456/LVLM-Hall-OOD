import os
from typing import Literal, Optional
import json
from tqdm import tqdm


class VQADataset():
    def __init__(self, annotation_path: str, data_folder: str, split: Literal['train', 'val']):
        self.ann_path = annotation_path
        self.img_root = data_folder
        self.split = split
        if not any([self.split == x for x in ['train', 'val']]):
            raise ValueError(f'split {self.split} no supported')
        self.data = self.get_data()

    def get_data(self) -> list:
        data = []
        with open(self.ann_path, 'r') as file:
            ann = json.load(file)
        data_cat = [
            {
                "img": os.path.join(self.img_root, f"COCO_{self.split}2014_{ins['image_id']:012}.jpg"),
                "question": f"{ins['question']}\nAnswer the question using a single word or phrase.\n",
                "gt_ans": ins['answers'],
                "question_id": ins["question_id"]
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
