import llama_iti
import numpy as np
import torch
from utils.tools import create_folder, get_short_name
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.arguments import Arguments
from llm.prompt_loader import PromptLoader


def llm_generate(dataset, used_indices):
    args = Arguments()
    cfg = args.get_config()

    num_gene = cfg.num_gene
    most_likely = cfg.most_likely
    model_name = cfg.model_name

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

    dataset_name = cfg.dataset_name

    if dataset_name == 'tydiqa':
        length = len(used_indices)
        indices = used_indices
    else:
        length = len(dataset)
        indices = [x for x in range(length)]

    create_folder(f'./save_for_eval/{dataset_name}_hal_det/')
    create_folder(f'./save_for_eval/{dataset_name}_hal_det/answers')

    prompt_loader = PromptLoader(dataset_name, dataset)

    for i in range(length):
        answers = [None] * num_gene

        prompt = prompt_loader.get_prompt(int(indices[i]))
        encoded_prompt = tokenizer(
            prompt, return_tensors='pt').input_ids.cuda()

        for gen_iter in range(num_gene):
            if most_likely:
                generated = model.generate(encoded_prompt,
                                           num_beams=5,
                                           num_return_sequences=1,
                                           do_sample=False,
                                           max_new_tokens=64,
                                           )
            else:
                generated = model.generate(encoded_prompt,
                                           do_sample=True,
                                           num_return_sequences=1,
                                           num_beams=1,
                                           max_new_tokens=64,
                                           temperature=0.5,
                                           top_p=1.0)

            decoded = tokenizer.decode(generated[0, encoded_prompt.shape[-1]:],
                                       skip_special_tokens=True)  # only get answers

            if dataset_name == 'tqa' or dataset_name == 'triviaqa':
                # corner case.
                if 'Answer the question concisely' in decoded:
                    print('#####error')
                    print(decoded.split('Answer the question concisely')[1])
                    print('#####error')
                    decoded = decoded.split('Answer the question concisely')[0]
            if dataset_name == 'coqa':
                if 'Q:' in decoded:
                    print('#####error')
                    print(decoded.split('Q:')[1])
                    print('#####error')
                    decoded = decoded.split('Q:')[0]

            answers[gen_iter] = decoded

        print('sample: ', i)
        if most_likely:
            info = 'most_likely_'
        else:
            info = 'batch_generations_'
        print("Saving answers")
        short_name = get_short_name(model_name)
        np.save(f'./save_for_eval/{dataset_name}_hal_det/answers/' + info + f'hal_det_{short_name}_{dataset_name}_answers_index_{i}.npy',
                {'prompt': prompt, 'answers': answers})
