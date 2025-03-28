import os
import numpy as np

from tqdm import tqdm
from typing import Literal, Optional
import pandas as pd
from datasets import load_dataset
from datasets import Dataset
import json


class AOKVQADataset():
    def __init__(self, split: Literal['train', 'validation']):
        '''Multiple-choices answering dataset'''
        self.split = split
        self.dataset = load_dataset('HuggingFaceM4/A-OKVQA', split=split)

    def get_data(self) -> Dataset:

        def transform(x):
            x['img'] = x['image']
            x['answer'] = x['direct_answers']
            x['question'] = f"{x['question']}\nAnswer the question using a single word or phrase.\n"
            return x

        data = self.dataset.map(transform, num_proc=8)
        cols_to_remove = data.column_names
        cols_to_remove.remove("img")
        cols_to_remove.remove("answer")
        cols_to_remove.remove("question")
        data = data.remove_columns(cols_to_remove)

        return data
