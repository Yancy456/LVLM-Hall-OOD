import os
import json

from dataset.base import BaseDataset
import datasets


class TriviaQA(BaseDataset):
    def __init__(self, prompter):
        super(TriviaQA, self).__init__()
        self.prompter = prompter

    def get_data(self):
        dataset = datasets.load_dataset(
            'TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']

        answerable = []  # answerable questions in dataset
        for x in dataset:
            if len(x['answers']['text']) == 0:
                continue

            answerable.append(self.phrase_question(x))
        return answerable, None

    def phrase_question(self, x):
        '''add prompt to question'''
        x['question'] = self.prompter.build_prompt(x['question'])
        x['answers'] = x['answers']['text']
        return x
