

class Pipeline:
    def __init__(self) -> None:
        pass

    def transform(self, X_train_original, X_test_original, y_train_original, y_test_original, pre_processor):
        X_train = pre_processor.train(X_train_original)
        X_test = pre_processor.transforms(X_test_original)
        y_train = y_train_original
        y_test = y_test_original

        return (
            X_train, X_test, y_train, y_test
        )
    