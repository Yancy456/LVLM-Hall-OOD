from sklearn.linear_model import LogisticRegression


class SemanticProb:
    def __init__(self) -> None:
        pass

    def train_single_metric(dataset, token_type='tbg', metric='b_entropy'):
        """train and test on single metric (e.g. SE, Acc) on single dataset"""
        var_name = token_type[0] + metric[0]
        # named as [te, se. ta, sa] for easy identification; t for tbg, s for slt (token positions)
        # e for entropy and a for accuracy (or model faithfulness)
        X_trains, X_vals, X_tests, y_trains, y_vals, y_tests = create_Xs_and_ys(
            getattr(D, f'{token_type}_dataset'), getattr(D, metric)
        )

        accs = []
        aucs = []
        models = []

        for i, (X_train, X_val, X_test, y_train, y_val, y_test) in enumerate(zip(X_trains, X_vals, X_tests, y_trains, y_vals, y_tests)):
            print(
                f"Training on {D.name}-{token_type.upper()}-{metric.upper()} {i+1}/{len(X_trains)}")
            model = LogisticRegression()
            sklearn_train_and_evaluate(model, X_train, y_train, X_val, y_val)
            test_loss, test_acc, test_auc = sklearn_evaluate_on_test(
                model, X_test, y_test)
            accs.append(test_acc)
            aucs.append(test_auc)
            models.append(model)

        setattr(D, f'{var_name}_accs', accs)
        setattr(D, f'{var_name}_aucs', aucs)
        setattr(D, f'{var_name}_models', models)

    def train_single_metric(D, token_type='tbg', metric='b_entropy'):
        """train and test on single metric (e.g. SE, Acc) on single dataset"""
        var_name = token_type[0] + metric[0]
        # named as [te, se. ta, sa] for easy identification; t for tbg, s for slt (token positions)
        # e for entropy and a for accuracy (or model faithfulness)
        X_trains, X_vals, X_tests, y_trains, y_vals, y_tests = create_Xs_and_ys(
            getattr(D, f'{token_type}_dataset'), getattr(D, metric)
        )

        accs = []
        aucs = []
        models = []

        for i, (X_train, X_val, X_test, y_train, y_val, y_test) in enumerate(zip(X_trains, X_vals, X_tests, y_trains, y_vals, y_tests)):
            print(
                f"Training on {D.name}-{token_type.upper()}-{metric.upper()} {i+1}/{len(X_trains)}")
            model = LogisticRegression()
            sklearn_train_and_evaluate(model, X_train, y_train, X_val, y_val)
            test_loss, test_acc, test_auc = sklearn_evaluate_on_test(
                model, X_test, y_test)
            accs.append(test_acc)
            aucs.append(test_auc)
            models.append(model)

        setattr(D, f'{var_name}_accs', accs)
        setattr(D, f'{var_name}_aucs', aucs)
        setattr(D, f'{var_name}_models', models)
