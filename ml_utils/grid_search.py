

class GridSearch:
    def __init__(self, evaluator: object, param_grid: dict, X_val, y_val) -> None:
        '''
        Grid search for two parameters

        param_grid={
            'param_name1':iterable,
            'param_name2':iterable
        }
        '''
        self.X_val = X_val
        self.y_val = y_val
        self.estimator = evaluator
        self.param_grid = param_grid

    def search(self):
        param1 = self.param.items()[0]
        param2 = self.param.items()[1]

        best_score = 0
        best_results = {}

        print(f'Grid Searching for best {param1['key']},{param2['key']}')

        for i in param1['value']:
            for j in param2['value']:
                score = self.estimator(i, j, self.X_val, self.y_val)
                if score > best_score:
                    best_score = score
                    best_results['param1'] = i
                    best_results['param2'] = j
        best_results['best_score'] = best_score
        return best_results
