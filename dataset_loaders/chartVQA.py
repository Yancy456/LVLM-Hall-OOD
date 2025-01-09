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
import re


class ChartVQADataset():
    def __init__(self, annotation_path: str, data_folder: str):
        self.ann_path = annotation_path
        self.img_root = data_folder

    def get_data(self) -> list:
        data = []
        with open(self.ann_path, 'r') as file:
            ann = json.load(file)
        data_cat = [
            {
                "img_path": os.path.join(self.img_root, f"{ins['imgname']}.png"),
                "question": f"{ins['query']}\nAnswer the question using a single word or digit number rather than spelling it out.\n",
                "answers": ins['label'],
                "question_id": ins["imgname"]
            }
            for ins in tqdm(ann)
        ]
        data += data_cat

        return data
