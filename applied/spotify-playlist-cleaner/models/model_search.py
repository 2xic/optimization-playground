from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn import svm
from .cosine_sim import CosineSim
from .torch_models.linear_torch_model import LinearTorchModel
from .torch_models.conv_torch_model import ConvTorchModel
from optimization_playground_shared.clustering.ElbowKmeans import ElbowKmeans

def find_model(X, y, x_test, y_test, log=True):
    best_model = None
    best_accuracy = 0

    for clf in [
        ConvTorchModel(len(X[0]), max(y) + 1),
        LinearTorchModel(len(X[0]), max(y) + 1),
        RandomForestClassifier(max_depth=22),
        RandomForestClassifier(max_depth=14),
        RandomForestClassifier(max_depth=12),
        RandomForestClassifier(max_depth=10),
        RandomForestClassifier(max_depth=8),
        RandomForestClassifier(max_depth=4),
        AdaBoostClassifier(n_estimators=100),
        svm.SVC(decision_function_shape='ovo'),
        GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=1.0,
            max_depth=1, 
            random_state=0
        ),
        CosineSim(),
        ElbowKmeans(),
    ]:
        clf.fit(X, y)
        accuracy = accuracy_score(clf.predict(x_test), y_test)
        if log:
            print(f"{clf.__class__.__name__} Accuracy {accuracy}")
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            best_model = clf
    return best_model, best_accuracy
