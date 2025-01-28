from .triviaQA_judge import TriviaQAJudge
# from .bleurt import TriviaQAHaloJudge
from .MMSafety_judge import MMSafetyJudge
from .ScienceQA import ScienceQAJudge
from .OKVQA import OKVQAJudge
from .ChartQA import ChartQAJudge
from .DocVQA import DocVQAJudge
from .InfoVQA import InfoVQAJudge


def load_judge(dataset, model_name=None):
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
    elif dataset == 'ChartQA':
        return ChartQAJudge()
    elif dataset == 'DocVQA':
        return DocVQAJudge(model_name)
    elif dataset == 'InfoVQA':
        return InfoVQAJudge()
    else:
        raise ValueError(f'no judge for {dataset}')
