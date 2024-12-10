from tqdm import tqdm
import torch
from llm_utils.llm_loader import load_llm
from utils.prompt import Prompter
from llm_utils.llm_generation import LLMGeneration
from utils.store_data import StoreData, ReadData
from utils.arguments import Arguments
from answer_judge import AnswerJudge
from dataset_loaders import load_data
from dataset_loaders.utils import data_sampler
import os


def main(args):
    # Load dataset
    prompter = Prompter(args.prompt, args.theme)

    # Load LLM and answer judge
    model, processor = load_llm(args.model_name, args.model_path)
    llm_generation = LLMGeneration(model, processor)
    judge = AnswerJudge(args.dataset, model_path=args.judge_path)
    store_data = StoreData(args.save_path)
    data = ReadData(args.save_path).read_all()

    # Generate responses and embeddings
    for ins in tqdm(data):

        if args.judge_type != 'no_judge':  # check answers
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


if __name__ == "__main__":
    arguments = Arguments()
    args = arguments.get_config()
    main(args)
