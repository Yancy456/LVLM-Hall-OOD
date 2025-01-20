

class OKVQAJudge:
    def __init__(self) -> None:
        pass

    def check(self, response: str, answer: int):
        letters = ['a', 'b', 'c', 'd', 'e']
        llm_answer = response.lower()[0]
        if llm_answer == letters[answer]:
            return 1
        else:
            return 0
