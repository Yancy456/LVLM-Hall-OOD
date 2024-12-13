import os
import json

from dataset.base import BaseDataset


class MMSafetyBench(BaseDataset):
    def __init__(self, prompter,  data_root: str, split="SD_TYPO"):
        super(MMSafetyBench, self).__init__()
        self.ann_root = data_root
        self.img_root = data_root
        self.split = split
        self.prompter = prompter

    def get_data(self):
        # All data in original MMSafetyBench are UNSAFE
        # In order to test performance of linear classification, author generates some safe data for validation

        # The original MMSafetyBench data
        # All data are unsafe queries.
        data = []
        val_phrases = []
        for sc in ['01-Illegal_Activitiy', '02-HateSpeech', '03-Malware_Generation', '04-Physical_Harm',
                   '05-EconomicHarm', '06-Fraud', '07-Sex', '08-Political_Lobbying',
                   '09-Privacy_Violence', '10-Legal_Opinion', '11-Financial_Advice',
                   '12-Health_Consultation', '13-Gov_Decision']:
            ann = json.load(
                open(os.path.join(self.ann_root, f"data/processed_questions/{sc}.json"), 'r'))
            val_phrases += [ins['Key Phrase'] for ins in ann.values()]
            print(sc)
            data = data + [
                {
                    "img_path": os.path.join(self.img_root, "data/imgs/", sc, self.split, f"{k}.jpg"),
                    "question": self.prompter.build_prompt(v['Rephrased Question']),
                    "is_safe": 0,  # 0 means unsafe, 1 means safe
                    "scenario": f"{sc}.json"
                }
                for k, v in ann.items()
            ]

        # Generated unsafe data for training
        sc = '00-Training'
        ann = json.load(open(os.path.join(
            self.ann_root, f"train_data/processed_questions/{sc}.json"), 'r'))
        print(sc)
        data = data + [
            {
                "img_path": os.path.join(self.img_root, "train_data/imgs/", sc, self.split, f"{k}.jpg"),
                "question": self.prompter.build_prompt(v['Rephrased Question']),
                "is_safe": 0,
                "scenario": f"{sc}.json"
            }
            for k, v in ann.items()
            if v['Key Phrase'] not in val_phrases
        ]

        # Generated safe data for both training and validation
        scenario_list = [
            "01-Daily_Activitiy",
            "02-Economics",
            "03-Physical",
            "04-Legal",
            "05-Politics",
            "06-Finance",
            "07-Health",
            "08-Sex",
            "09-Government",
        ]
        for sc in scenario_list:
            ann = json.load(open(os.path.join(
                self.ann_root, "safe_data/processed_questions/", f"{sc}.json")))
            data = data + [
                {
                    "img_path": os.path.join(self.img_root, "safe_data/imgs/",  sc, self.split, f"{k}.jpg"),
                    "question": self.prompter.build_prompt(v['Rephrased Question']),
                    "is_safe": 1,
                    "scenario": f"{sc}.json"
                }
                for k, v in ann.items()
                if v['Key Phrase'] not in val_phrases
            ]
        print(len(data))
        return data
