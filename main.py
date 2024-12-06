from tqdm import tqdm
import torch
from llm_utils.llm_loader import load_llm
from utils.prompt import Prompter
from llm_utils.llm_generation import LLMGeneration
from utils.store_data import StoreData
from utils.arguments import Arguments
from answer_judge import AnswerJudge
from dataset_loaders import load_data
from dataset_loaders.utils import data_sampler


def main(args):

    # Load dataset
    prompter = Prompter(args.prompt, args.theme)
    data = load_data(args.dataset, prompter,
                     args.annotation_path, args.data_folder, args.split, args.category)
    if args.num_samples is not None:
        data = data_sampler(
            data, num_samples=args.num_samples, shuffle=args.shuffle)

    # Load LLM and answer judge
    model, processor = load_llm(args.model_name, args.model_path)
    llm_generation = LLMGeneration(model, processor)
    judge = AnswerJudge(args.dataset, model_path=args.judge_path)
    store_data = StoreData(args.save_path)

    # Generate responses and embeddings
    for ins in tqdm(data):
        img_path = ins['img_path'] if 'img_path' in ins else None
        results = llm_generation.generate(
            prompt=ins['question'], img_path=img_path, hidden_state_type='post-generation')

        ins.update(results)

        if args.judge_type != 'no_judge':  # check answer
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
    arguments = Arguments(
        default_config='/home/hallscope/configs/pope/val_popular.yaml')
    args = arguments.get_config()
    main(args)
