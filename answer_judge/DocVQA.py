from transformers import AutoModelForCausalLM, AutoTokenizer


class DocVQAJudge:
    def __init__(self, model_name: str) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto").cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def check(self, response: str, answer: list[str]):
        temp = f"Is response '%s' equivalent to ground truth '%s'? Respond with either 'Yes' or 'No' only."

        for ans in answer:
            prompt = temp % (response.lower(), ans.lower())
            inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
            outputs = self.model.generate(**inputs)
            entailment = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True).lower()
            if entailment == 'yes':
                return 1

        return 0
