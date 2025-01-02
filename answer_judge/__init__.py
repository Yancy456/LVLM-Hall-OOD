from .triviaQA_judge import TriviaQAJudge
# from .bleurt import TriviaQAHaloJudge
from .MMSafety_judge import MMSafetyJudge
from .ScienceQA import ScienceQAJudge
from .OKVQA import OKVQAJudge


def load_judge(dataset, prompt_type=None):
    if dataset == 'MMSafety':
        return MMSafetyJudge(prompt_type)
    elif dataset == 'trivia_qa':
        return TriviaQAJudge()
    elif dataset == 'trivia_qa_halo':
        return TriviaQAHaloJudge(self.model_path)
    elif dataset == 'ScienceQA':
        return ScienceQAJudge()
    elif dataset == 'OKVQA':
        return OKVQAJudge()
    else:
        raise ValueError(f'no judge for {dataset}')
