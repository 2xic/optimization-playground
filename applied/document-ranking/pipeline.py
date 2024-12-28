

class Pipeline:
    def __init__(self, sample_size=1) -> None:
        self.sample_size = sample_size
        assert 0 < self.sample_size and self.sample_size <= 1

    def transform(self, X_train_original, X_test_original, y_train_original, y_test_original, pre_processor):
        print(type(X_train_original))
        X_train_original = X_train_original[:max(1, int(len(X_train_original) * self.sample_size))]
        X_test_original = X_test_original[:max(1, int(len(X_test_original) * self.sample_size))]
        y_train_original = y_train_original[:max(1, int(len(y_train_original) * self.sample_size))]
        y_test_original = y_test_original[:max(1, int(len(y_test_original) * self.sample_size))]

        if (callable(pre_processor)):
            pre_processor = pre_processor()

        X_train = pre_processor.fit_transforms(X_train_original)
        X_test = pre_processor.transforms(X_test_original)
        y_train = y_train_original
        y_test = y_test_original

        return (
            X_train, X_test, y_train, y_test
        )
