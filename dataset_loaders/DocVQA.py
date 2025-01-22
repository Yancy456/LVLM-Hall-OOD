import os
import numpy as np

from typing import Literal, Optional
import pandas as pd
from datasets import load_dataset
from datasets import Dataset
from tqdm import tqdm


class DocVQADataset():
    def __init__(self, split: Literal['validation', 'test']):
        '''Multiple-choices answering dataset'''
        self.dataset = load_dataset(
            'lmms-lab/DocVQA', name='DocVQA', split=split)

    def get_data(self) -> list:
        data = [
            {
                "img": ins['image'],
                "question": f"{ins['question']}\nAnswer the question using a single word or digit number.\n",
                "answer": ins['answers']
            }
            for ins in tqdm(self.dataset)
        ]
        return data
