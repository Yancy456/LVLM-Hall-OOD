import os
import argparse
import time

import cv2
import json
import numpy as np
from tqdm import tqdm

from model import build_model
from dataset import build_dataset
from utils.func import split_list, get_chunk
from utils.prompt import Prompter
from utils.store_data import StoreData
import torch
from utils.arguments import Arguments


def get_model_output(args, data, model, extra_keys):
    store_data = StoreData(args.save_path)

    for ins in tqdm(data):
        img_id = ins['img_path'].split("/")[-1]
        if args.model_name == "GPT4V":
            image = [ins['img_path']]
            prompt = ins['question']
            response_text = model.forward(image, prompt)
            out = {
                "image": img_id,
                "question": prompt,
                "label": ins["label"],
                "response": response_text,
            }
            print(response_text)
            for key in extra_keys:
                out[key] = ins[key]
            ans_file.write(json.dumps(out) + "\n")
        else:
            image = cv2.imread(ins['img_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            prompt = ins['question']
            response_text, output_ids, logits, probs, hidden_states = model.forward_with_probs(
                image, prompt)

            # only using last generation results
            hidden_states = hidden_states[-1]
            hidden_states = torch.stack(hidden_states, dim=0).squeeze(
                dim=1)  # stack all layers into one dimension

            # shape=(layers,1,n_hidden)
            hidden_states = hidden_states[:, -1:, :].detach().cpu().numpy()

            if len(logits) <= args.token_id:
                continue
            out = {
                "image": img_id,
                "model_name": args.model_name,
                "question": prompt,
                "label": ins["label"],
                "response": response_text,
                "output_ids": output_ids.tolist(),
                "logits": logits.tolist()[args.token_id],
                'hidden_states': hidden_states
            }
            for key in extra_keys:
                out[key] = ins[key]
            store_data.store(out)


def main(args):
    model = build_model(args)
    prompter = Prompter(args.prompt, args.theme)

    data, extra_keys = build_dataset(args.dataset, args.split, prompter)
    if args.num_samples is not None:
        if args.sampling == 'first':
            data = data[:args.num_samples]
        elif args.sampling == "random":
            np.random.shuffle(data)
            data = data[:args.num_samples]
        else:
            labels = np.array([ins['label'] for ins in data])
            classes = np.unique(labels)
            data = np.array(data)
            final_data = []
            for cls in classes:
                cls_data = data[labels == cls]
                idx = np.random.choice(
                    range(len(cls_data)), args.num_samples, replace=False)
                final_data.append(cls_data[idx])
            data = list(np.concatenate(final_data))

    print(args.num_chunks, args.chunk_idx)
    data = get_chunk(data, args.num_chunks, args.chunk_idx)

    if not os.path.exists(f"./output/{args.model_name}/"):
        os.makedirs(f"./output/{args.model_name}/")

    results = get_model_output(
        args, data, model, extra_keys)


if __name__ == "__main__":
    arguments = Arguments('./configs/mad_val.yaml')
    args = arguments.get_config()

    main(args)
