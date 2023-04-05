from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

def find_model(X, y, x_test, y_test):
    for clf in [
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
        )
    ]:
        clf.fit(X, y)
        accuracy = accuracy_score(clf.predict(x_test), y_test)
        print(f"Accuracy {accuracy}")
