from tqdm import tqdm


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
        self.evaluator = evaluator
        self.param_grid = param_grid

    def search(self):
        params = list(self.param_grid.items())
        param1 = params[0]
        param2 = params[1]

        best_score = 0
        best_results = {}

        print(f"Grid Searching for best {param1[0]},{param2[0]}")

        for i in tqdm(param1[1]):
            for j in tqdm(param2[1], leave=False):
                score = self.evaluator(i, j, self.X_val, self.y_val)
                if score > best_score:
                    print(
                        f"found best {param1[0]}:{i},{param2[0]}:{j} Score: {score}")
                    best_score = score
                    best_results[param1[0]] = i
                    best_results[param2[0]] = j
        best_results['best_score'] = best_score
        return best_results
