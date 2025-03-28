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

    def multi_generate(self, batch_prompts: List[str], batch_imgs: Tensor, args):
        config = {
            'max_new_tokens': 50,
            'do_sample': False,
            # 'stop_strings': ['\n'],
            'return_dict_in_generate': True,
            'output_hidden_states': True,
            'output_scores': False,
            'num_beams': args.num_beams,
            'output_logits': args.return_logits,
        }

        inputs, prompts = self.encode_prompts(batch_prompts, batch_imgs)

        outputs = self.model.generate(
            **inputs, **config)

        hidden_states = self.phrase_hidden_states(outputs.hidden_states)

        most_likely_response = self.processor.batch_decode(
            outputs.sequences, skip_special_tokens=True)
        most_likely_response = self.phrase_responses(
            most_likely_response, prompts, 1)

        '''
        outputs.logits=Tuple(logits1,tensor2,...). Tuple contains each generation
        logits1.shape=(batch,num_vocabulary)
        '''
        if args.return_logits:
            most_likely_logits = outputs.logits[0]
        else:
            most_likely_logits = None

        all_responses = None
        if args.n_multi_gene > 0:
            '''whether to generate more than one answers, typically used for Semantic Analysis'''
            config = {
                'max_new_tokens': 50,
                'do_sample': True,
                'temperature': 1.0,
                'num_return_sequences': args.n_multi_gene,
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
                all_responses, prompts, args.n_multi_gene)

        return {
            "most_likely": {
                'embedding': hidden_states,
                'response': most_likely_response,
                'logits': most_likely_logits
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
            if len(prompts) == 1:
                inputs = self.processor(images=images[0], text=prompts[0],
                                        return_tensors='pt', padding=True, do_rescale=False).to(0, torch.float16)
            else:
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
            prompt = prompts[i]
            prompt = prompt.replace("<image>", " ")
            x = x[len(prompt):]
            x = x.replace("<pad>", "").replace(
                "</s>", "").replace("<unk>", "").replace("<|im_end|>", "").replace("<image>", " ").strip()
            return x

        rsps = []
        for i, x in enumerate(responses):
            # divide n_multi_gene to process batch data
            x = phraser(i//n_multi_gene, x)
            rsps.append(x)
        return rsps
