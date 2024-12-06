import torch
from transformers import TextStreamer
from typing import Literal
import cv2


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

    def generate(self, prompt, img_path=None, hidden_state_type: Literal['post-generation', 'SLT'] = 'SLT'):
        config = {
            'max_new_tokens': 50,
            # 'stop_strings': ['\n'],
            'return_dict_in_generate': True,
            'output_hidden_states': True if hidden_state_type != 'post_generation' else False,
            'output_scores': False,
            'output_logits': False,
        }
        inputs = self.encode_prompts(prompt, img_path)

        outputs = self.model.generate(
            **inputs, **config)

        if hidden_state_type != 'post-generation':
            hidden_states = self.phrase_hidden_states(outputs.hidden_states)
        elif hidden_state_type == 'post-generation':
            with torch.no_grad():
                hidden_states = self.model(
                    outputs.sequences[:, :-1], output_hidden_states=True).hidden_states
                hidden_states = torch.stack(hidden_states, dim=0).squeeze()
                hidden_states = hidden_states.detach().cpu().numpy()[:, -1, :]
        else:
            raise ValueError('hidden_state_type error')

        input_ids = inputs['input_ids']
        most_likely_response = self.processor.decode(
            outputs.sequences[0, input_ids.shape[1]:], skip_special_tokens=True)
        most_likely_response = self.phrase_responses(most_likely_response)

        return {
            "most_likely": {
                'embedding': hidden_states,
                'response': most_likely_response
            }
        }

    def encode_prompts(self, prompt, img_path):
        if img_path != None:
            # Vision Model
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"}
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True)

            inputs = self.processor(images=image, text=prompt,
                                    return_tensors='pt', padding=True).to(0, torch.float16)
            return inputs
        else:
            # Language Model
            inputs = self.processor(prompt, return_tensors="pt").to(0)
            return inputs

    def phrase_hidden_states(self, hidden_states, token_position: str = 'SLT'):
        '''
        outputs.hidden_states : Tuple1(Tuple2(hidden_state_tensor,...),...), the Tuple1 contains hidden_states of each generation. The Tuple2 contains hidden_states of each layer.
        hidden_state_tensor.shape=(1,query_len=700,hidden_size=4096)

        outputs.scores : Tuple(score_tensor,...), a Tuple contains score tensors of all output tokens. 
        score_tensor.shape=(1,vacabulary_size)
        '''
        if token_position == 'SLT':
            # only using last generation results
            hidden_states = hidden_states[-2]
            hidden_states = torch.stack(hidden_states, dim=0).squeeze(
                dim=1)  # stack all layers into one dimension

            # shape=(layers,1,n_hidden)
            hidden_states = hidden_states[:, -1:,
                                          :].detach().cpu().float().numpy()
        return hidden_states

    def phrase_responses(self, responeses):
        '''remove reluctant words'''
        def phraser(x):
            return x.replace("<pad>", "").replace("</s>", "").replace("<unk>", "").strip().lower()

        return list(map(phraser, responeses)) if isinstance(responeses, list) else phraser(responeses)

    def set_config(self, **new_config):
        '''set new config for generation'''
        new_cfg = self.generation_cfg.copy(
        )  # Copy the old dictionary new_dict.update(additional_keys)
        new_cfg.update(new_config)
        return new_cfg
