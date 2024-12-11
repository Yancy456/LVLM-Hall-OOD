import torch
from transformers import TextStreamer
from typing import Literal, List, Dict
import cv2
from datasets import Dataset
from PIL import Image


class LLMGeneration():
    def __init__(self, model, processor) -> None:
        self.model = model
        self.processor = processor

        self.generation_cfg = {  # default generation configuration
            'max_new_tokens': 50,
            'stop_strings': ['\n'],

            # sampling strategy
            'do_sample': True,
            'top_p': 1.0,
            'temperature': 0.1,

            # return formation
            'return_dict_in_generate': True,
            'output_hidden_states': True,
            'output_scores': False,
            'output_logits': False
        }

    def generate(self, batch: Dataset, hidden_state_type: Literal['SLT'] = 'SLT'):
        config = {
            'max_new_tokens': 50,
            # 'stop_strings': ['\n'],
            'return_dict_in_generate': True,
            'output_hidden_states': True,
            'output_scores': False,
            'output_logits': False,
        }
        inputs, prompts = self.encode_prompts(batch)

        outputs = self.model.generate(
            **inputs, **config)

        hidden_states = self.phrase_hidden_states(outputs.hidden_states)

        most_likely_response = self.processor.batch_decode(
            outputs.sequences, skip_special_tokens=True)
        most_likely_response = self.phrase_responses(
            most_likely_response, prompts)

        return {
            "most_likely": {
                'embedding': hidden_states,
                'response': most_likely_response
            }
        }

    def encode_prompts(self, batch: Dataset):
        if 'img_path' in batch:
            # Vision Model
            img_paths = batch['img_path']
            images = [Image.open(path) for path in img_paths]

            def apply_to_template(prompt):
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image"}
                        ],
                    }]
                return self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True)

            prompts = [apply_to_template(x) for x in batch['question']]

            inputs = self.processor(images=images, text=prompts,
                                    return_tensors='pt', padding=True).to(0, torch.float16)
            return inputs, prompts
        else:
            # Language Model
            prompts = batch['question']
            inputs = self.processor(prompts, return_tensors="pt").to(0)
            return inputs, prompts

    def phrase_hidden_states(self, hidden_states, token_position: str = 'SLT'):
        '''
        outputs.hidden_states : Tuple1(Tuple2(hidden_state_tensor,...),...), the Tuple1 contains hidden_states of each generation. The Tuple2 contains hidden_states of each layer.
        hidden_state_tensor.shape=(batch,query_len=700,hidden_size=4096)
        '''
        if token_position == 'SLT':
            # only using second last generation results
            hidden_states = hidden_states[-2]
            # stack all layers into one dimension
            hidden_states = torch.stack(hidden_states, dim=0)

            # shape=(layers,batch_size,query_len,n_hidden)
            hidden_states = hidden_states[:, :, -1,
                                          :].detach().cpu().transpose(0, 1).float().numpy()

            # shape=(batch_size,layers,n_hidden)
        return hidden_states

    def phrase_responses(self, responeses: List[str], prompts: List[str]) -> List[str]:
        '''remove reluctant words and only keep respones without prompts'''
        def phraser(i, x):
            prompt = prompts[i].replace('<image>', ' ')
            x = x[len(prompt):]
            x = x.replace("<pad>", "").replace(
                "</s>", "").replace("<unk>", "").strip()
            return x

        rsps = []
        for i, x in enumerate(responeses):
            x = phraser(i, x)
            rsps.append(x)
        return rsps

    def set_config(self, **new_config):
        '''set new config for generation'''
        new_cfg = self.generation_cfg.copy(
        )  # Copy the old dictionary new_dict.update(additional_keys)
        new_cfg.update(new_config)
        return new_cfg
