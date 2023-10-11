# Import necessary libraries
import time
import numpy as np
import ml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a scikit-learn DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# Measure inference time without Hummingbird
start_time = time.time()
y_pred_sklearn = clf.predict(X_test)
end_time = time.time()
sklearn_inference_time = end_time - start_time

print(f"y_pred_sklearn: #{y_pred_sklearn}")

# Convert the scikit-learn model to a Hummingbird format
hb_model = ml.convert(clf, 'ivy')

# Measure inference time with Hummingbird
start_time = time.time()
y_pred_hb = hb_model.predict(X_test)
end_time = time.time()
hb_inference_time = end_time - start_time

print(f"y_pred_hb: {y_pred_hb}")

# Evaluate the model
#from sklearn.metrics import accuracy_score
# accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
# accuracy_hb = accuracy_score(y_test, y_pred_hb)

# print(f"Scikit-learn Inference Time: {sklearn_inference_time} seconds")
# print(f"Hummingbird Inference Time: {hb_inference_time} seconds")

# print(f"Scikit-learn Accuracy: {accuracy_sklearn}")
# print(f"Hummingbird Accuracy: {accuracy_hb}")
