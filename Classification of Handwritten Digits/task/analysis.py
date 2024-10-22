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
from sklearn.model_selection import train_test_split, GridSearchCV

def fit_predict_eval(model, features_train, features_test, target_train, target_test):

    model.fit(features_train, target_train)

    predictions = model.predict(features_test)

    score = accuracy_score(target_test, predictions)

    #print(f'Model: {model.__class__.__name__}')
    #print(f'Accuracy: {score:.4f}\n')
    return score

(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train[:6000]
y_train = y_train[:6000]

x_train_flattened = x_train.reshape(x_train.shape[0], -1)

x_train_part, x_test_part, y_train_part, y_test_part = train_test_split(x_train_flattened, y_train, test_size=0.3, random_state=40)

normalizer = Normalizer()
x_train_norm = normalizer.fit_transform(x_train_part)
x_test_norm = normalizer.transform(x_test_part)

knn_param_grid = {
    'n_neighbors': [3, 4],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'brute']
}

rf_param_grid = {
    'n_estimators': [300, 500],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample'],
    'random_state': [40]
}

knn_model = KNeighborsClassifier()
knn_grid_search = GridSearchCV(estimator=knn_model, param_grid=knn_param_grid, scoring='accuracy', n_jobs=-1)
knn_grid_search.fit(x_train_norm, y_train_part)

rf_model = RandomForestClassifier()
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(x_train_norm, y_train_part)

best_knn = knn_grid_search.best_estimator_
knn_accuracy = fit_predict_eval(best_knn, x_train_norm, x_test_norm, y_train_part, y_test_part)

best_rf = rf_grid_search.best_estimator_
rf_accuracy = fit_predict_eval(best_rf, x_train_norm, x_test_norm, y_train_part, y_test_part)


print(f"K-nearest neighbours algorithm\nbest estimator: {best_knn}\naccuracy: {knn_accuracy:.4f}\n")
print(f"Random forest algorithm\nbest estimator: {best_rf}\naccuracy: {rf_accuracy:.4f}")