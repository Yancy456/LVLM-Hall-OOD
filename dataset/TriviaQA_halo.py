import os
import json

from dataset.base import BaseDataset
import datasets
from datasets import load_dataset


class TriviaQAHalo(BaseDataset):
    def __init__(self, prompter):
        super(TriviaQAHalo, self).__init__()
        self.prompter = prompter

    def get_data(self):
        dataset = self.load_triviaqa()

        answerable = []  # answerable questions in dataset
        for x in dataset:
            # if len(x['answers']['text']) == 0:
            #    continue

            answerable.append(self.phrase_question(x))
        return answerable

    def phrase_question(self, x):
        '''add prompt to question'''
        y = {}
        y['question'] = self.prompter.build_prompt(x['question'])
        y['answers'] = x['answer']['aliases']
        y['question_id'] = x['question_id']
        return y

    def load_triviaqa(self):
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        id_mem = set()

        def remove_dups(batch):
            if batch['question_id'][0] in id_mem:
                return {_: [] for _ in batch.keys()}
            id_mem.add(batch['question_id'][0])
            return batch

        dataset = dataset.map(remove_dups, batch_size=1,
                              batched=True, load_from_cache_file=False)
        return dataset
