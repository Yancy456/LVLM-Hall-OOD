import torch
from transformers import TextStreamer
from typing import Literal, List, Dict
import cv2
from datasets import Dataset
from PIL import Image
from torch import Tensor


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

    def generate(self, batch_prompts: List[str], batch_imgs: Tensor, hidden_state_type: Literal['SLT'] = 'SLT'):
        config = {
            'max_new_tokens': 50,
            'do_sample': False,
            # 'stop_strings': ['\n'],
            'return_dict_in_generate': True,
            'output_hidden_states': True,
            'output_scores': False,
            'output_logits': False,
        }
        inputs, prompts = self.encode_prompts(batch_prompts, batch_imgs)

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

    def multi_generate(self, batch_prompts: List[str], batch_imgs: Tensor, n_multi_gene: int = 0, hidden_state_type: Literal['SLT'] = 'SLT'):
        config = {
            'max_new_tokens': 50,
            'do_sample': False,
            # 'stop_strings': ['\n'],
            'return_dict_in_generate': True,
            'output_hidden_states': True,
            'output_scores': False,
            'output_logits': False,
        }
        inputs, prompts = self.encode_prompts(batch_prompts, batch_imgs)

        outputs = self.model.generate(
            **inputs, **config)

        hidden_states = self.phrase_hidden_states(outputs.hidden_states)

        most_likely_response = self.processor.batch_decode(
            outputs.sequences, skip_special_tokens=True)
        most_likely_response = self.phrase_responses(
            most_likely_response, prompts, 1)

        all_responses = None
        if n_multi_gene > 0:
            '''whether to generate more than one answers, typically used for Semantic Analysis'''
            config = {
                'max_new_tokens': 50,
                'do_sample': True,
                'temperature': 1.0,
                'num_return_sequences': n_multi_gene,
                'return_dict_in_generate': True,
                'output_hidden_states': False,
                'output_scores': False,
                'output_logits': False,
            }

            outputs = self.model.generate(
                **inputs, **config)
            all_responses = self.processor.batch_decode(
                outputs.sequences, skip_special_tokens=True)
            all_responses = self.phrase_responses(
                all_responses, prompts, n_multi_gene)

        return {
            "most_likely": {
                'embedding': hidden_states,
                'response': most_likely_response
            },
            'responses': all_responses
        }

    def encode_prompts(self, batch_prompts: List[str], batch_imgs: Tensor):
        if batch_imgs != None:
            # Vision Model
            images = batch_imgs

            def apply_to_template(prompt):
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ],
                    }]
                return self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True)

            prompts = [apply_to_template(x) for x in batch_prompts]
            # WARNING: do_rescale MUST be False, because transforms.ToTensor() already done that
            inputs = self.processor(images=images, text=prompts,
                                    return_tensors='pt', padding=True, do_rescale=False).to(0, torch.float16)
            return inputs, prompts
        else:
            # Language Model
            prompts = batch_prompts
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

    def phrase_responses(self, responses: List[str], prompts: List[str], n_multi_gene: int) -> List[str]:
        '''remove reluctant words and only keep response without prompts'''
        def phraser(i, x):
            prompt = prompts[i].replace('<image>', ' ')
            x = x[len(prompt):]
            x = x.replace("<pad>", "").replace(
                "</s>", "").replace("<unk>", "").strip()
            return x

        rsps = []
        for i, x in enumerate(responses):
            # divide n_multi_gene to process batch data
            x = phraser(i//n_multi_gene, x)
            rsps.append(x)
        return rsps

    def set_config(self, **new_config):
        '''set new config for generation'''
        new_cfg = self.generation_cfg.copy(
        )  # Copy the old dictionary new_dict.update(additional_keys)
        new_cfg.update(new_config)
        return new_cfg
