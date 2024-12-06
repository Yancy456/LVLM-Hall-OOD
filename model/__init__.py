

from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import torch
from PIL import Image
import requests


def build_model(args):
    if args.model_name == "InstructBLIP":
        from .InstructBLIP import InstructBLIP
        model = InstructBLIP(args)
    elif args.model_name == "LLaVA-7B-new":
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
        processor = AutoProcessor.from_pretrained(
            args.model_path, trust_remote_code=True)
        return model, processor
    elif args.model_name == "LLaMa-7B":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
        processor = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True)
        return model, processor

    elif args.model_name == "LLaVA-7B":
        from .LLaVA import LLaVA
        model = LLaVA(args)
        return model
    elif args.model_name == "LLaVA-13B":
        pass
        # model=AutoModel.from_pretrained(args.model_path,torch_dtype=torch.float16,trust_remote_code=True).cuda()
        # tokenizer = AutoTokenizer.from_pretrained(
        # args.model_path, trust_remote_code=True)

    elif args.model_name == "LLaMA_Adapter":
        from .LLaMA_Adapter import LLaMA_Adapter
        model = LLaMA_Adapter(args)
    elif args.model_name == "MMGPT":
        from .MMGPT import MMGPT
        model = MMGPT(args)
    elif args.model_name == "GPT4V":
        from .GPT4V import GPTClient
        model = GPTClient()
    elif args.model_name == "MiniGPT4":
        from .MiniGPT4 import MiniGPT4
        model = MiniGPT4(args)
    elif args.model_name == "mPLUG-Owl":
        from .mPLUG_Owl import mPLUG_Owl
        model = mPLUG_Owl(args)
    else:
        model = None

    return model
