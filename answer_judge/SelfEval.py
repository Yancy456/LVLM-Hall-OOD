

class SelfEvalJudge:
    def __init__(self) -> None:
        pass

    def check(self, response: str, answer: int):
        llm_answer = response.lower()[0]
        if answer == 1 and llm_answer == 'a':
            return 1
        elif answer == 0 and llm_answer == 'b':
            return 1
        else:
            return 0
