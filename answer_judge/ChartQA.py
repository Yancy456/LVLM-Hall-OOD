

class ChartQAJudge:
    def __init__(self) -> None:
        pass

    def check(self, response: str, answer: str):
        def process_str(s):
            try:
                # Attempt to convert the string to a float
                return float(s)
            except ValueError:
                # If conversion fails, return the string in lowercase
                return s.lower()

        if process_str(response) == process_str(answer):
            return 1
        else:
            return 0
