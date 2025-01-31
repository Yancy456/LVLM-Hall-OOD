import os
import numpy as np

from typing import Literal, Optional
import json
from tqdm import tqdm


class VQAIDKDataset():
    def __init__(self, annotation_path: str, data_folder: str):
        self.ann_path = annotation_path
        self.img_root = data_folder

    def get_data(self) -> list:
        data = []
        with open(self.ann_path, 'r') as file:
            ann = json.load(file)
        data_cat = [
            {
                "img_path": os.path.join(self.img_root, ins['image']),
                "question": f"Given the question '{ins['question']}', is the question answerable or unanswerable based on the image?\nPlease reply with 'Unanswerable' or 'Answerable'. \n",
                "answer": 'unanswerable' if 'keywords' in ins else 'answerable',
            }
            for ins in tqdm(ann)
        ]
        data += data_cat

        return data
