from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from optimization_playground_shared.apis.url_to_text import get_document_eval
from optimization_playground_shared.utils.ClassImbalanceSplitter import balance_classes
from sklearn import svm
import torch

"""
Metrics for knowing if the model embedding even learn anything.
"""
class EvaluationMetrics:
    def __init__(self):
        X, y = get_document_eval()
        X, y = balance_classes(X, y)    
        X = [str(i) for i in X]
        assert type(X[0]) == str, type(X[0])
        self.X_train_original, self.X_test_original, self.y_train_original, self.y_test_original = train_test_split(
            X, y, test_size=0.33, random_state=42
        )

    def eval(self, embedding_model):
        model = svm.SVR()
        with torch.no_grad():
            (X_train, X_test, y_train, y_test) = (
                embedding_model.transforms(self.X_train_original).cpu(), 
                embedding_model.transforms(self.X_test_original).cpu(), 
                self.y_train_original, 
                self.y_test_original,
            )        
            model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, list(map(lambda x: (1 if x > 0.5 else 0), model.predict(X_test))))
            return accuracy
