import os
import argparse
import time

import cv2
import json
import numpy as np
from tqdm import tqdm
import torch
from model import build_model
from dataset import build_dataset
from utils.func import split_list, get_chunk
from utils.prompt import Prompter
from llm_utils.llm_generation import LLMGeneration
from utils.store_data import StoreData
from utils.arguments import Arguments
from ML_tools.semantic_clustering.semantic_entropy import EntailmentDeberta
from ML_tools.semantic_clustering.semantic_clustering import SemanticClustering
from answer_judge import AnswerJudge
from dataset_loaders import load_data
from dataset_loaders.utils import data_sampler


def get_model_output(args, data, model, processor):
    ''''
    data=[{
    'img_path':the path of input image
    'question': the question
    'examples':[a list of example answers]
    '...': other dataset specific keys
    },...]
    '''

    store_data = StoreData(args.save_path)
    llm_generation = LLMGeneration(model, processor)
    judge = AnswerJudge(args.dataset, model_path=args.judge_path)
    for ins in tqdm(data):
        # Vision model or Language model
        if 'img_path' in ins:
            image = cv2.imread(ins['img_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ins['question']},
                        {"type": "image"}
                    ],
                }
            ]
            prompt = processor.apply_chat_template(
                conversation, add_generation_prompt=True)

            inputs = processor(images=image, text=prompt,
                               return_tensors='pt', padding=True).to(0, torch.float16)
            results = llm_generation.generate(
                **inputs)
        else:
            image = None
            prompt = processor(ins['question'], return_tensors="pt").to(0)[
                'input_ids']
            results = llm_generation.generate(hidden_state_type='post-generation',
                                              input_ids=prompt)

        ins.update(results)

        if args.judge_type != 'no_judge':
            label = judge.check_answer(ins)
            ins['label'] = label

        '''
        ins={
        'img_path':the path of input image
        'question': the question
        'answers':[a list of example answers]
        'most_likely':{
            'response': the most likely response
            'embedding': hidden_states
        },
        'responses':[other generation sequences],
        'label':boolean
        '...': other dataset specific keys
        }
        '''

        store_data.store(ins)


def main(args):
    model, processor = build_model(args)
    prompter = Prompter(args.prompt, args.theme)

    data = build_dataset(args.dataset, args.split, prompter)

    if args.num_samples is not None:
        data = data_sampler(
            data, num_samples=args.num_samples, shuffle=args.shuffle)

    get_model_output(
        args, data, model, processor)


if __name__ == "__main__":
    arguments = Arguments(default_config='./configs/trivia_halo_qa.yaml')
    args = arguments.get_config()
    main(args)
