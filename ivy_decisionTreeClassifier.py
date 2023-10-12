from sklearn.datasets import load_breast_cancer
import ivy.functional.frontends.sklearn as sklearn_frontend
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# uncomment this to test with real-life data and comment the below data generation procedure
# X, y = load_breast_cancer(return_X_y=True)
# X = X[40:60, :]
# y = y[40:60]

num_classes = 3
X = np.random.rand(10, 5)
y = np.random.randint(num_classes, size=10)


sk_clf = DecisionTreeClassifier(max_depth=3)
ivy_clf = sklearn_frontend.tree.DecisionTreeClassifier(max_depth=3)

sk_clf.fit(X, y)
ivy_clf.fit(X, y)

sk_y = sk_clf.predict(X)
ivy_y = ivy_clf.predict(X)

print("ivy-based prediction: ", ivy_y)
print("sklearn-based prediction: ", sk_y)
