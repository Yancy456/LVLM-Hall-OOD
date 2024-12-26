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


class VizWizDataset():
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
                "question": self.prompter(ins['question']),
                "answer_type": ins['answer_type'],
                "answers": ins['answers']
            }
            for ins in tqdm(ann)
        ]
        data += data_cat

        return data

    def prompter(self, question: str, prompt_type: Literal['answerable', 'open_end'] = 'answerable'):
        answerable_prompt = "Given the question '%s', is the question answerable or unanswerable based on the image?\nPlease reply with 'Unanswerable' or 'Answerable'."
        open_end_prompt = '%s\nAnswer the question using a single word or phrase.\n'
        if prompt_type == 'answerable':
            return answerable_prompt % question
        elif prompt_type == 'open_end':
            return open_end_prompt % question
        else:
            raise ValueError(f'prompt type {prompt_type} no supported')
