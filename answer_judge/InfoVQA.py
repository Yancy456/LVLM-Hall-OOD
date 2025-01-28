

class InfoVQAJudge:
    def __init__(self) -> None:
        pass

    def check(self, response: str, answer: str):
        def process_str(s):
            s = s.strip('.').strip()
            return s.lower()

        if process_str(response) == process_str(answer):
            return 1
        else:
            return 0
