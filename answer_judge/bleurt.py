from evaluate import load
import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
import numpy as np


class BleurtJudge:
    def __init__(self, bleurt_path) -> None:
        self.model = BleurtForSequenceClassification.from_pretrained(
            bleurt_path).cuda()
        self.tokenizer = BleurtTokenizer.from_pretrained(bleurt_path)
        self.model.eval()

    def check_answer(self, responses: list, answers: list, scores: float = 0.5):
        predictions = [s.lower() for s in responses]
        all_answers = answers
        all_results = np.zeros((len(all_answers), len(predictions)))
        with torch.no_grad():
            for anw in range(len(all_answers)):
                inputs = self.tokenizer(predictions, [all_answers[anw]] * len(predictions),
                                        padding='longest', return_tensors='pt')
                for key in list(inputs.keys()):
                    inputs[key] = inputs[key].cuda()
                res = np.asarray(self.model(
                    **inputs).logits.flatten().tolist())
                all_results[anw] = res
        score = np.max(all_results)
        label = 1 if score > 0.5 else 0
        return label
