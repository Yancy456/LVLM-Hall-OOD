import os

import numpy as np

from dataset.base import BaseDataset
import json
from typing import Literal, Optional
import pandas as pd
from datasets import Dataset
from tqdm import tqdm


class SelfEvalDataset():
    def __init__(self, annotation_path: str):
        self.ann_path = annotation_path

    def selfeval_prompt(self, question, response):
        prompt = f'Question: {question}\n Proposed Answer: {response}\n Is the proposed answer: A.True \nB.False\n'
        return prompt

    def get_data(self) -> list:
        with open(self.ann_path) as file:
            ann = json.load(file)
        data_cat = [
            {
                "img_path": ins['img_path'],
                "question": self.selfeval_prompt(ins['question'], ins['response']),
                "is_correct": ins['is_correct'],  # 1 yes; 0 no
            }
            for ins in ann
        ]

        data = []
        for i in tqdm(range(len(data_cat))):  # check image existence
            if os.path.isfile(data_cat[i]['img_path']):
                data.append(data_cat[i])
        return data
