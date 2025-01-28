import os
import numpy as np

from tqdm import tqdm
from typing import Literal, Optional
import pandas as pd
from datasets import load_dataset
from datasets import Dataset
import json


class InfoVQADataset():
    def __init__(self):
        '''Multiple-choices answering dataset'''
        self.dataset = load_dataset('vidore/infovqa_train', split='train')

    def get_data(self) -> Dataset:

        def transform(x):
            x['img'] = x['image']
            x['answer'] = x['answer']
            x['question'] = f"{x['query']}\nAnswer the question using a single word or phrase.\n"
            return x

        data = self.dataset.map(transform, num_proc=8)
        cols_to_remove = data.column_names
        cols_to_remove.remove("img")
        cols_to_remove.remove("answer")
        cols_to_remove.remove("question")
        data = data.remove_columns(cols_to_remove)

        return data
