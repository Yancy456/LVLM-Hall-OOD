import os

import numpy as np

from typing import Literal, Optional
import pandas as pd
from datasets import load_dataset


class ScienceQA():
    def __init__(self, split: Literal['train', 'val', 'test']):
        '''Multiple-choices answering dataset'''
        self.dataset = load_dataset('derek-thomas/ScienceQA', split=split)

    def prompter(self, question: str, answer: list):
        choices = ' '.join([f'{i}.{q}' for i, q in enumerate(answer)])
        return prompt % (question, choices)

    def get_data(self) -> list:
        data = [
            {
                "img_path": ins['image'],
                "question": self.prompter(ins['question'], ins['choices']),
                "answer": ins['answer'],
                "category": ins['subject']
            }
            for ins in self.dataset
        ]
        return data


prompt = '''Question:%s
Choices: %s
'''
