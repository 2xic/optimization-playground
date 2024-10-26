import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from src.regression.linear_regression import ProbabilisticLinearRegression
from src.regression.huber_regression import HuberRegression
import os
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

# Our models from mini sklearn
linear_regression = ProbabilisticLinearRegression()
huber_regression = HuberRegression()
sklearn_regression = SklearnLinearRegression()

def predict(model, x, y, plotLine):
    a = plotLine.reshape((-1)).reshape(-1, 1)
    if isinstance(model, SklearnLinearRegression):
        model.fit(x, y)        
        output = model.predict(a)
    else:
        model.fit(x, y, n_iterations=10_000)
        output = model.predict(a.reshape(-1).tolist())
    print(output)
    return output

# Got idea from
# https://developer.nvidia.com/blog/dealing-with-outliers-using-three-robust-linear-regression-models/
N_SAMPLES = 50
N_OUTLIERS = 0

X, y, coef = datasets.make_regression(
	n_samples=N_SAMPLES,
	n_features=1,
	n_informative=1,
	noise=2,
	coef=True,
	random_state=42
)
coef_list = [
    ["original_coef", float(coef)]
]
print(coef_list)
# add outliers          	 
np.random.seed(42)
X[:N_OUTLIERS] = 10 + 0.75 * np.random.normal(size=(N_OUTLIERS, 1))
y[:N_OUTLIERS] = -15 + 20 * np.random.normal(size=N_OUTLIERS)

#plt.scatter(X, y)
#plt.savefig("regression.png")

predict_X = np.arange(X.min(), X.max()).reshape(-1, 1)

fit_df = pd.DataFrame(
	index = predict_X.flatten(),
	data={
        "linear_regression": predict(linear_regression, X, y, predict_X),
	    "huber_regression": predict(huber_regression, X, y, predict_X),
	 #   "sklearn_regression": predict(sklearn_regression, X, y, predict_X),
	}
)
fix, ax = plt.subplots()
fit_df.plot(ax=ax)
plt.scatter(X, y, c="k")
plt.title("Linear regression on data with outliers")
plt.savefig(os.path.join(
    os.path.dirname(__file__),
    "regressions.png"
))
