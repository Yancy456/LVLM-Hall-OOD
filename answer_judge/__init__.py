from .triviaQA_judge import TriviaQAJudge
from .triviaQA_halo_judge import TriviaQAHaloJudge


class AnswerJudge:
    def __init__(self, dataset, prompt_type=None, model=None, tokenizer=None, model_path=None, judge_type='self') -> None:
        self.dataset = dataset
        self.prompt_type = prompt_type
        self.model_path = model_path

        self.judge = self.get_judge(dataset, prompt_type, judge_type)

    def get_judge(self, dataset, prompt_type, judge_type):
        if dataset == 'MMSafety':
            return
        if dataset == 'trivia_qa':
            return TriviaQAJudge()
        if dataset == 'trivia_qa_halo':
            return TriviaQAHaloJudge(self.model_path)

    def check_answer(self, instance: object):
        return self.judge.check_answer(instance)
