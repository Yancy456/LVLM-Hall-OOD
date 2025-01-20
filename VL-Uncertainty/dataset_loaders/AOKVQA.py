import os
import numpy as np

from tqdm import tqdm
from typing import Literal, Optional
import pandas as pd
from datasets import load_dataset
from datasets import Dataset
import json


class AOKVQADataset():
    def __init__(self, annotation_path: str, data_folder: StopIteration):
        '''Multiple-choices answering dataset'''
        self.ann_path = annotation_path
        self.data_folder = data_folder

    def prompter(self, question: str, answer: list):
        letters = ['A', 'B', 'C', 'D', 'E']
        choices = ' '.join([f'{letters[i]}.{q}' for i, q in enumerate(answer)])
        return prompt % (question, choices)

    def get_data(self) -> list:

        with open(self.ann_path, 'r') as file:
            ann = json.load(file)
        data_cat = [
            {
                "img_path": os.path.join(self.data_folder, f"{ins['image_id']:012}.jpg"),
                "question": self.prompter(ins['question'], ins['choices']),
                "answer": ins['correct_choice_idx']
            }
            for ins in tqdm(ann)
        ]

        data = []
        for i in tqdm(range(len(data_cat))):  # check image existence
            if os.path.isfile(data_cat[i]['img_path']):
                data.append(data_cat[i])

        return data


prompt = '''Question:%s
Answer the Question with following choices.
Choices: %s
'''
