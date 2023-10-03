#import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier

num_classes = 3
X = np.random.rand(10, 5)
y = np.random.randint(num_classes, size=10)
# Train a scikit-learn DecisionTreeClassifier
print("---sample_decisionTreeClassifier.py---")
print(f"X: {X}")
print(f"y: {y}")
print("---sample_decisionTreeClassifier.py---")
clf = DecisionTreeClassifier(max_depth=3)
print("---sample_decisionTreeClassifier.py---")
print(f"clf: {clf}")
print("---sample_decisionTreeClassifier.py---")
#clf.fit(X, y)

# Measure inference time without Hummingbird
# start_time = time.time()
# y_pred_sklearn = clf.predict(X)
# end_time = time.time()
# sklearn_inference_time = end_time - start_time
