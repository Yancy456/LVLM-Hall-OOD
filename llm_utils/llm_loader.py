from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


def load_llm(model_name, model_path):
    if model_name == 'LLaVA':
        processor = LlavaNextProcessor.from_pretrained(model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda:0")

        return model, processor
    elif model_name == "LLaMa-7B":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
        processor = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        return model, processor

    elif model_name == "QWen2-VL":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True)
        return model, processor
    else:
        raise ValueError(f'No such model {model_name}')
