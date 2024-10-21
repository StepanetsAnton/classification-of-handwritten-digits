import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer

def fit_predict_eval(model, features_train, features_test, target_train, target_test):

    model.fit(features_train, target_train)

    predictions = model.predict(features_test)

    score = accuracy_score(target_test, predictions)

    print(f'Model: {model.__class__.__name__}')
    print(f'Accuracy: {score:.4f}\n')
    return score

(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train[:6000]
y_train = y_train[:6000]

x_train_flattened = x_train.reshape(x_train.shape[0], -1)

x_train_part, x_test_part, y_train_part, y_test_part = train_test_split(x_train_flattened, y_train, test_size=0.3, random_state=40)

normalizer = Normalizer()
x_train_norm = normalizer.fit_transform(x_train_part)
x_test_norm = normalizer.transform(x_test_part)

models = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(random_state=40),
    LogisticRegression(max_iter=1000, random_state=40),
    RandomForestClassifier(random_state=40)
]

accuracies = []
for model in models:
    accuracy = fit_predict_eval(model, x_train_norm, x_test_norm, y_train_part, y_test_part)
    accuracies.append((model.__class__.__name__, accuracy))

accuracies_sorted = sorted(accuracies, key=lambda x: x[1], reverse=True)

improvement = any(acc > 0.9 for _, acc in accuracies_sorted)

print(f"The answer to the 1st question: {'yes' if improvement else 'no'}")
print(f"The answer to the 2nd question: {accuracies_sorted[0][0]}-{accuracies_sorted[0][1]:.3f}, "
      f"{accuracies_sorted[1][0]}-{accuracies_sorted[1][1]:.3f}")