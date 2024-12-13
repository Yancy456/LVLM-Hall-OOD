from .triviaQA_judge import TriviaQAJudge
from .triviaQA_halo_judge import TriviaQAHaloJudge
from .MMSafety_judge import MMSafetyJudge


def load_judge(dataset, prompt_type):
    if dataset == 'MMSafety':
        return MMSafetyJudge(prompt_type)
    elif dataset == 'trivia_qa':
        return TriviaQAJudge()
    elif dataset == 'trivia_qa_halo':
        return TriviaQAHaloJudge(self.model_path)
    else:
        raise ValueError(f'no judge for {dataset}')
