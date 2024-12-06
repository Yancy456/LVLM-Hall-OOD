from evaluate import load


class TriviaQAJudge:
    def __init__(self) -> None:
        pass

    def check_answer(self, instance):
        '''using Stanford Question Answering Dataset (SQuAD) metric'''
        squad_metric = load("squad_v2")
        response: str = instance['most_likely']['response']
        examples: list = instance['answers']

        def metric(response: str, examples: list, *args, **kwargs):
            '''construct squad metric format
            Example format:
            predictions=[{'prediction_text': 'salesman',
                'no_answer_probability': 0.0, 'id': 'qb_5618--167/167_422354.txt#0_0'}]
            references=[{'answers': {'answer_start': [351], 'text': [
                'butcher']}, 'id': 'qb_5618--167/167_422354.txt#0_0'}]
            '''
            prediction = {'prediction_text': response,
                          'no_answer_probability': 0.0, 'id': '0'}

            reference = {'answers': {'answer_start': [0],
                                     'text': examples
                                     }, 'id': '0'}

            results = squad_metric.compute(
                predictions=[prediction],
                references=[reference])
            return 1.0 if (results['f1'] >= 50.0) else 0.0

        return metric(response, examples)
