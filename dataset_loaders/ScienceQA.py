import os

import numpy as np

from typing import Literal, Optional
import pandas as pd
from datasets import load_dataset
from datasets import Dataset


class ScienceQA():
    def __init__(self, split: Literal['train', 'val', 'test']):
        '''Multiple-choices answering dataset'''
        self.dataset = load_dataset('derek-thomas/ScienceQA', split=split)

    def prompter(self, question: str, answer: list):
        letters = ['A', 'B', 'C', 'D', 'E']
        choices = ' '.join([f'{letters[i]}.{q}' for i, q in enumerate(answer)])
        return prompt % (question, choices)

    def get_data(self) -> Dataset:
        def transform_example(ins):
            return {
                "img": ins['image'],
                "question": self.prompter(ins['question'], ins['choices']),
                "answer": ins['answer']
            }

        # Use map to apply the transformation to the dataset
        transformed = self.dataset.map(transform_example, num_proc=8, remove_columns=[
                                       'choices', 'solution', 'task', 'lecture', 'skill', 'task', 'hint', 'grade', 'topic', 'subject', 'image', 'category'])
        return transformed


prompt = '''Question:%s
Answer the Question with following choices.
Choices: %s
'''
