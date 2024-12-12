from tqdm import tqdm
import torch
from llm_utils.llm_loader import load_llm
from utils.prompt import Prompter
from llm_utils.llm_generation import LLMGeneration
from utils.store_data import StoreData
from utils.arguments import Arguments
from answer_judge import AnswerJudge
from dataset_loaders import load_data, ImageDataset
from dataset_loaders.utils import data_sampler
import os
from datasets import Dataset
from torch.utils.data import DataLoader


def main(args):
    # Load dataset
    prompter = Prompter(args.prompt, args.theme)
    data = load_data(args.dataset, prompter,
                     args.annotation_path, args.data_folder, args.split, args.batch_size, args.category)  # type: Dataset
    if args.num_samples is not None:
        data = data_sampler(
            data, num_samples=args.num_samples, shuffle=args.shuffle)
    data_loader = DataLoader(
        data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Load LLM and answer judge
    model, processor = load_llm(args.model_name, args.model_path)
    llm_generation = LLMGeneration(model, processor)
    judge = AnswerJudge(args.dataset, model_path=args.judge_path)
    store_data = StoreData(args.save_path)

    # Generate responses and embeddings
    for batch in tqdm(data_loader):
        results = llm_generation.generate(batch)

        batch.update(results)
        batch['label'] = batch['label'].numpy()

        # if args.judge_type != 'no_judge':  # check answers
        #    label = judge.check_answer(batch)
        #    batch['label'] = label

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
        store_data.store(batch)


if __name__ == "__main__":
    arguments = Arguments('./configs/pope/train_popular.yaml')
    args = arguments.get_config()
    main(args)
