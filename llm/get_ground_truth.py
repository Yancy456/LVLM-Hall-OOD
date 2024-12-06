import numpy as np
import evaluate
import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from utils.tools import get_short_name
from utils.arguments import Arguments
from tqdm import tqdm
import os


def get_ground_truth(dataset, used_indices):
    args = Arguments()
    cfg = args.get_config()
    bleurt_path = cfg.bleurt_path

    most_likely = cfg.most_likely
    dataset_name = cfg.dataset_name
    model_name = cfg.model_name
    short_name = get_short_name(model_name)
    use_rouge = cfg.use_rouge

    model = BleurtForSequenceClassification.from_pretrained(bleurt_path).cuda()
    tokenizer = BleurtTokenizer.from_pretrained(bleurt_path)
    model.eval()

    rouge = evaluate.load('rouge')
    gts = np.zeros(0)
    if dataset_name == 'tydiqa':
        length = len(used_indices)
    else:
        length = len(dataset)
    for i in tqdm(range(length)):
        if dataset_name == 'tqa':
            best_answer = dataset[i]['best_answer']
            correct_answer = dataset[i]['correct_answers']
            all_answers = [best_answer] + correct_answer
        elif dataset_name == 'triviaqa':
            all_answers = dataset[i]['answer']['aliases']
        elif dataset_name == 'coqa':
            all_answers = dataset[i]['answer']
        elif dataset_name == 'tydiqa':
            all_answers = dataset[int(used_indices[i])]['answers']['text']

        if most_likely:
            if not os.path.exists(f'./save_for_eval/{dataset_name}_hal_det/answers/most_likely_hal_det_{short_name}_{dataset_name}_answers_index_{i}.npy'):
                break
            answers = np.load(
                f'./save_for_eval/{dataset_name}_hal_det/answers/most_likely_hal_det_{short_name}_{dataset_name}_answers_index_{i}.npy')
        else:
            answers = np.load(
                f'./save_for_eval/{dataset_name}_hal_det/answers/batch_generations_hal_det_{short_name}_{dataset_name}_answers_index_{i}.npy')

        # get the ground truth answers.
        if use_rouge:
            predictions = answers
            all_results = np.zeros((len(all_answers), len(predictions)))
            all_results1 = np.zeros((len(all_answers), len(predictions)))
            all_results2 = np.zeros((len(all_answers), len(predictions)))
            for anw in range(len(all_answers)):
                results = rouge.compute(predictions=predictions,
                                        references=[all_answers[anw]
                                                    ] * len(predictions),
                                        use_aggregator=False)
                all_results[anw] = results['rougeL']
                all_results1[anw] = results['rouge1']
                all_results2[anw] = results['rouge2']

            # breakpoint()
            gts = np.concatenate([gts, np.max(all_results, axis=0)], 0)
        else:  # using BLEURT metric
            predictions = answers
            all_results = np.zeros((len(all_answers), len(predictions)))
            with torch.no_grad():
                for anw in range(len(all_answers)):
                    inputs = tokenizer(predictions, [all_answers[anw]] * len(predictions),
                                       padding='longest', return_tensors='pt')
                    for key in list(inputs.keys()):
                        inputs[key] = inputs[key].cuda()
                    res = np.asarray(model(**inputs).logits.flatten().tolist())
                    all_results[anw] = res
            gts = np.concatenate([gts, np.max(all_results, axis=0)], 0)

    # breakpoint()
    if most_likely:
        if use_rouge:
            np.save(f'./ml_{dataset_name}_rouge_score.npy', gts)
        else:
            np.save(f'./ml_{dataset_name}_bleurt_score.npy', gts)
    else:
        if use_rouge:
            np.save(f'./bg_{dataset_name}_rouge_score.npy', gts)
        else:
            np.save(f'./bg_{dataset_name}_bleurt_score.npy', gts)
