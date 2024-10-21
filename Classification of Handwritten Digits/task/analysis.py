import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train[:6000]
y_train = y_train[:6000]

x_train_flattened = x_train.reshape(x_train.shape[0], -1)

x_train_part, x_test_part, y_train_part, y_test_part = train_test_split(x_train_flattened, y_train, test_size=0.3, random_state=40)

print(f"x_train shape: {x_train_part.shape}")
print(f"x_test shape: {x_test_part.shape}")
print(f"y_train shape: {y_train_part.shape}")
print(f"y_test shape: {y_test_part.shape}")

class_distribution = pd.Series(y_train_part).value_counts(normalize=True)
print("\nProportion of samples per class in train set:")
print(class_distribution)